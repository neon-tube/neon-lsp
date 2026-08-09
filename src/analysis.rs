//! Running the front end for an editor rather than for a build.
//!
//! The CLI's `check` renders diagnostics to stderr and calls `exit(1)` at the first stage
//! that fails, which is right for a build and useless here: an editor needs the errors as
//! *data*, and needs them for a file that does not compile — that is the only time anyone
//! is looking. So this walks the same pipeline and collects instead of exiting.
//!
//! It stops at the first stage that produces errors, deliberately. Parse errors make the
//! AST a guess, and type errors derived from a guessed AST are noise that sends people
//! chasing problems they do not have. One real error beats twenty invented ones.
//!
//! **What this module keeps, and why.** It used to return `Vec<Diagnostic>` and drop
//! everything else — including the `TypecheckResult`, which it had just spent the whole
//! check computing. That map holds every expression's type and every name's definition
//! site, which is to say it holds hover, jump-to-definition, find-references, rename,
//! completion detail, inlay hints and signature help. All of it was being discarded
//! microseconds after it was built, and the module doc in `main.rs` used to describe
//! go-to-definition as blocked on "a span-to-definition index" that in fact already
//! existed. So `analyze` now returns a `Checked` and the server holds onto it. Nothing
//! about the check itself got slower; the result simply stopped being thrown away.

use crate::position::LineIndex;
use neon_compiler::typecheck::env::Unit;
use neon_compiler::typecheck::result::TypecheckResult;
use neon_compiler::typecheck::Env;
use neon_compiler::{ast, expand, lexer, parser, stdlib};
use std::ops::Range;
use std::path::{Path, PathBuf};

/// One diagnostic, in byte offsets. Converting to LSP's line/column pairs is the
/// protocol layer's job — this stays in the compiler's own coordinate system so nothing
/// here has to think about UTF-16.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub span: Range<usize>,
    pub message: String,
    /// Extra spans that explain the error, each with its own note. Rendered as related
    /// information so an editor can offer to jump to them.
    pub labels: Vec<(Range<usize>, String)>,
    pub help: Option<String>,
    pub severity: Severity,
    /// The lint name for a warning — the compiler's typed `Lint` spelled out — so the
    /// editor can show a code and a client can key quick-fixes (`@allow(...)`) off it.
    pub code: Option<&'static str>,
    /// A mechanical fix, when the checker computed one. Typed rather than parsed back
    /// out of the help text: the compiler's `Suggestion` is the single source and this
    /// is its editor-facing face.
    pub fix: Option<Fix>,
}

/// What a quick-fix would do.
#[derive(Debug, Clone)]
pub enum Fix {
    /// Insert `use <path>;` with the document's other imports.
    InsertUse(String),
}

/// Error or warning, the only two severities the compiler produces. Its own enum rather
/// than `lsp_types::DiagnosticSeverity` so this module keeps compiling without the
/// protocol crate in scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

impl Diagnostic {
    fn plain(span: Range<usize>, message: String) -> Self {
        Diagnostic {
            span,
            message,
            labels: Vec::new(),
            help: None,
            severity: Severity::Error,
            code: None,
            fix: None,
        }
    }
}

/// A file taking part in a compilation, with what an editor needs to point into it.
///
/// A `DefSite` names a module and a byte range, and byte ranges are meaningless without
/// the text they index. This is the other half: it turns "module `std::io`, bytes 412..418"
/// into a file URI and a line/column range. The `LineIndex` is built once per file rather
/// than per jump, because a jump into the stdlib would otherwise re-scan the file for
/// newlines every time the user pressed the key.
pub struct Source {
    pub module: Vec<String>,
    /// Where it lives on disk. Stdlib files always have one; the document being edited
    /// has its own URI already and never comes from here.
    pub path: PathBuf,
    /// Owns the file's text; `index.text()` is the source a span indexes into.
    pub index: LineIndex,
    /// The lexer's trivia table, kept solely for doc comments — `///` text is lexed and
    /// then dropped by the parser, since the AST has no field for it, so the only way to
    /// answer "what does this stdlib function document itself as" is to go back to the
    /// trivia and find the run of `Doc` comments ending where the declaration begins.
    pub trivia: Vec<lexer::Trivia>,
}

/// The stdlib, parsed once and reused for every check in the session.
///
/// **Why this cache is sound.** The obvious worry is two compilations sharing mutable
/// state and one leaking into the other's diagnostics. They cannot here, because nothing
/// downstream can mutate these modules: `Env::build_with` and `check::check_all` both take
/// `&[(Vec<String>, &ast::Module)]`, the AST has no interior mutability, and the `Env` that
/// accumulates every bit of inference is built fresh per check and dropped with it. The
/// cached value is immutable input, not carried-over results.
///
/// The `ExprId` numbering is the other half. `parse_from(sources, 0)` numbers the stdlib
/// `0..next_id` and reports `next_id`; each check then renumbers the *user's* fresh module
/// from that same base. So every check sees exactly the id assignment an uncached run
/// would have produced — the numbering is a pure function of the sources, and the sources
/// are toolchain data that cannot change while the server is running. A stdlib that does
/// change means a toolchain that changed, which means a restart.
///
/// What is *not* cached is the `Env`. Declaration and body resolution are global and
/// ordered across all modules at once (a stdlib fn may name a user type), so the stdlib's
/// half of an `Env` is not separable from the user's. Caching it would mean sharing
/// inference state between compilations, which is the unsound thing this comment opens by
/// ruling out. Parsing is the part that is genuinely independent, so parsing is the part
/// that is cached.
struct Cached {
    modules: Vec<(Vec<String>, ast::Module)>,
    next_id: u32,
    /// Parallel to `modules`, and in the same order: `stdlib::parse_from` preserves the
    /// order of the sources it was handed, which is what makes the two zippable and is
    /// the whole reason a jump into the stdlib can name a file at all.
    sources: Vec<Source>,
}

/// A completed check, kept so the editor can ask questions about it.
///
/// The three pieces are inseparable in practice: `result` is keyed by `ExprId` and says
/// nothing about where those ids are in the text, `module` carries the spans that answer
/// that, and `env` is what turns a `TyId` back into something printable. A query needs all
/// three, so they travel together.
pub struct Checked {
    /// The analyzed document's module, expanded and numbered — the same AST the checker
    /// saw, so an `ExprId` from `result` indexes into it. In a project this is the OPEN
    /// document's module, which need not be the entry.
    pub module: ast::Module,
    pub result: TypecheckResult,
    pub env: Env,
    /// The other project files of this run — entry and sibling modules, never the
    /// analyzed document itself — so a jump or hover into `util::helper` can open
    /// `src/util.neon` exactly as one into `io::println` opens the stdlib's file.
    /// Empty outside a project.
    pub sources: Vec<Source>,
}

impl Checked {
    /// The file a module was declared in: this run's project files first, the session's
    /// stdlib second. One lookup for every feature, so none of them can know about only
    /// half the world.
    ///
    /// Exact match first; failing that, the LONGEST non-empty prefix — a declaration
    /// inside `internal mod raw` of `std::string` carries the module
    /// `["std","string","raw"]`, and the file that holds it is `std/string.neon`. The
    /// entry's empty path is never used as a prefix: it would claim every module in
    /// existence, and "no file" (meaning: the document being edited) is the honest
    /// answer when nothing longer matched.
    pub fn source_of<'a>(
        &'a self,
        analyzer: &'a Analyzer,
        module: &[String],
    ) -> Option<&'a Source> {
        let all = || self.sources.iter().chain(analyzer.sources());
        all().find(|s| s.module == module).or_else(|| {
            all()
                .filter(|s| !s.module.is_empty() && module.starts_with(&s.module))
                .max_by_key(|s| s.module.len())
        })
    }
}

/// The front end, bound to one session's stdlib.
pub struct Analyzer {
    /// `None` when the toolchain could not be found. The server tells the user about that
    /// separately; see `main.rs`.
    cached: Option<Cached>,
    config: expand::Config,
}

/// Everything one run of the front end produced.
///
/// `checked` is `None` whenever the file did not reach a clean parse, which is the
/// ordinary state of a file mid-keystroke. The server keeps the last non-`None` one and
/// answers hover and jumps from that, because a hover that blanks out every time a
/// half-typed line fails to parse is a hover nobody trusts.
pub struct Analysis {
    /// The analyzed document's own diagnostics.
    pub diagnostics: Vec<Diagnostic>,
    /// Diagnostics belonging to OTHER project files, one entry per file whether it has
    /// any or not — publishing an empty list is how a fixed error leaves the editor's
    /// problems panel. The text is the text the spans index, carried so the server can
    /// turn byte offsets into positions without re-reading a file that may have moved
    /// on. Empty outside a project.
    pub foreign: Vec<(PathBuf, String, Vec<Diagnostic>)>,
    pub checked: Option<Checked>,
}

/// A project as the analyzer sees it: the manifest's directory and every `src/**/*.neon`.
struct Layout {
    /// `(src-relative path, absolute path)`, sorted; the entry `main.neon` included.
    files: Vec<(String, PathBuf)>,
}

/// The project containing `doc`, found the way the CLI finds it: walk up to a
/// `neon.toml`, take `src/` beneath it. `None` — no manifest, or the document is not
/// under that project's `src/` — means single-file analysis, which is what a scratch
/// buffer wants.
fn discover(doc: &Path) -> Option<Layout> {
    let mut dir = doc.parent();
    let root = loop {
        let d = dir?;
        if d.join("neon.toml").is_file() {
            break d;
        }
        dir = d.parent();
    };
    let src = root.join("src");
    doc.strip_prefix(&src).ok()?;
    let mut files = Vec::new();
    collect_neon(&src, &src, &mut files);
    files.sort();
    Some(Layout { files })
}

fn collect_neon(src_root: &Path, dir: &Path, out: &mut Vec<(String, PathBuf)>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for path in entries.flatten().map(|e| e.path()) {
        if path.is_dir() {
            collect_neon(src_root, &path, out);
        } else if path.extension().is_some_and(|e| e == "neon") {
            let rel = path
                .strip_prefix(src_root)
                .expect("collected under src")
                .to_string_lossy()
                .replace('\\', "/");
            out.push((rel, path));
        }
    }
}

/// The module a project file declares: the entry is the root (anonymous) module, every
/// other file is named by its path — `util.neon` is `util` — the same rule the stdlib
/// follows, because it is the same `module_path`.
fn module_of(rel: &str) -> Vec<String> {
    if rel == "main.neon" {
        Vec::new()
    } else {
        stdlib::module_path(rel)
    }
}

impl Analyzer {
    /// Parse the stdlib once. An unparseable stdlib is a broken toolchain rather than a
    /// user error, so it is reported as such and the server continues syntax-only.
    pub fn new(dir: &Path, std_sources: &[(String, String)]) -> Result<Analyzer, String> {
        let (modules, next_id) = stdlib::parse_from(std_sources, 0)?;

        // Zip rather than re-derive: `parse_from` computed each module's path from its
        // relative filename and returned them in order, so pairing by index recovers the
        // filename without asking the compiler to hand back something it already used.
        let sources = modules
            .iter()
            .zip(std_sources)
            .map(|((path, _), (rel, text))| {
                let lexed = lexer::lex_full(text);
                Source {
                    module: path.clone(),
                    path: dir.join(rel),
                    index: LineIndex::new(text),
                    // A stdlib that did not lex would have failed `parse_from` above, so
                    // this cannot be `Err` — but an empty trivia table degrades to
                    // "no documentation", which is the right way to be wrong.
                    trivia: lexed.map(|l| l.trivia).unwrap_or_default(),
                }
            })
            .collect();

        Ok(Analyzer {
            cached: Some(Cached {
                modules,
                next_id,
                sources,
            }),
            config: config(),
        })
    }

    /// Lexer and parser diagnostics only, for a session with no usable toolchain.
    ///
    /// The checker is skipped rather than run against an empty stdlib on purpose: with no
    /// `std` in scope every single name in a normal file is undefined, and the resulting
    /// wall of red says nothing true about the user's code.
    pub fn syntax_only() -> Analyzer {
        Analyzer {
            cached: None,
            config: config(),
        }
    }

    /// The stdlib files, for turning a `DefSite` into a location. Project files are
    /// per-run and live on `Checked::sources`; `Checked::source_of` is the lookup that
    /// consults both.
    pub fn sources(&self) -> &[Source] {
        self.cached
            .as_ref()
            .map(|c| c.sources.as_slice())
            .unwrap_or_default()
    }

    /// Everything the front end can say about one file, in pipeline order.
    ///
    /// `doc_path` is where the document lives, when it lives anywhere: it is how the
    /// analyzer finds the project around it. `overlays` are the other OPEN documents'
    /// current texts — the editor's copy is authoritative, and a sibling module open in
    /// the next split may be several keystrokes ahead of its file on disk.
    pub fn analyze(
        &self,
        doc_path: Option<&Path>,
        src: &str,
        overlays: &[(PathBuf, String)],
    ) -> Analysis {
        let bail = |d: Vec<Diagnostic>| Analysis {
            diagnostics: d,
            foreign: Vec::new(),
            checked: None,
        };

        let tokens = match lexer::lex(src) {
            Ok(t) => t,
            Err(errors) => {
                return bail(
                    errors
                        .iter()
                        .map(|e| Diagnostic::plain(e.span.clone(), e.to_string()))
                        .collect(),
                )
            }
        };

        let (module, errors) = parser::parse(&tokens, src.len());
        if !errors.is_empty() {
            return bail(
                errors
                    .iter()
                    .map(|e| Diagnostic::plain(e.span.clone(), e.to_string()))
                    .collect(),
            );
        }
        let Some(mut module) = module else {
            return bail(Vec::new());
        };

        let (expanded, _meta, expand_errors) = expand::expand(module, &self.config);
        if !expand_errors.is_empty() {
            return bail(
                expand_errors
                    .iter()
                    .map(|e| Diagnostic::plain(e.span.clone(), e.message.clone()))
                    .collect(),
            );
        }
        module = expanded;

        let Some(cached) = &self.cached else {
            return bail(Vec::new());
        };

        // The project around the document, when there is one. Each sibling file — the
        // entry included — becomes the module its path names, its text taken from an
        // open buffer when the editor has one and from disk otherwise. A sibling that
        // does not parse is SKIPPED, not fatal: its own window is already showing its
        // errors, and this document's analysis is worth more degraded than absent.
        let doc = doc_path.and_then(|p| std::fs::canonicalize(p).ok());
        let layout = doc.as_ref().and_then(|p| discover(p));
        let mut doc_module: Vec<String> = Vec::new();
        let mut sibling_modules: Vec<(Vec<String>, ast::Module)> = Vec::new();
        let mut sources: Vec<Source> = Vec::new();
        if let (Some(layout), Some(doc)) = (&layout, &doc) {
            for (rel, abs) in &layout.files {
                let canonical = std::fs::canonicalize(abs).unwrap_or_else(|_| abs.clone());
                if &canonical == doc {
                    doc_module = module_of(rel);
                    continue;
                }
                let text = match overlays.iter().find(|(p, _)| p == &canonical) {
                    Some((_, t)) => t.clone(),
                    None => match std::fs::read_to_string(abs) {
                        Ok(t) => t,
                        Err(_) => continue,
                    },
                };
                let Some(m) = parse_quietly(&text, &self.config) else {
                    continue;
                };
                let trivia = lexer::lex_full(&text).map(|l| l.trivia).unwrap_or_default();
                sibling_modules.push((module_of(rel), m));
                sources.push(Source {
                    module: module_of(rel),
                    path: abs.clone(),
                    index: LineIndex::new(&text),
                    trivia,
                });
            }
        }

        // Numbering: the stdlib first, siblings after it, this document last — every
        // `ExprId` in the compilation unique, so one `TypecheckResult` covers all of it.
        let mut next_id = cached.next_id;
        for (_, m) in &mut sibling_modules {
            next_id = ast::number_exprs_from(m, next_id);
        }
        ast::number_exprs_from(&mut module, next_id);

        let mut modules: Vec<(Vec<String>, &ast::Module)> =
            cached.modules.iter().map(|(p, m)| (p.clone(), m)).collect();
        modules.extend(sibling_modules.iter().map(|(p, m)| (p.clone(), m)));
        modules.push((doc_module.clone(), &module));

        // `RootApplication` because an editor is nearly always looking at a program: it is
        // the stricter of the two (a library has no `main` to demand), so this errs toward
        // showing a diagnostic rather than hiding one.
        let mut env = Env::build_with(&modules, Unit::RootApplication);
        if !env.errors().is_empty() {
            let errs = env.take_errors();
            let (own, foreign) = route(
                errs.iter().map(|e| (e.module.clone(), convert(e))),
                &doc_module,
                &sources,
            );
            return Analysis {
                diagnostics: own,
                foreign: with_text(foreign, &sources),
                checked: None,
            };
        }

        // Every diagnostic of the run, resolution errors included: `check_all` drains the
        // environment's channel into what it returns, so an unknown type written inside a
        // body reaches the editor rather than vanishing.
        //
        // An error belongs to the module it was raised in. This document's land here; a
        // sibling project file's are routed to that file (the editor shows them on it,
        // open or not); a stdlib module's are dropped — anchored here they would
        // underline whatever token sits at the same byte offset, and a broken stdlib is
        // a broken toolchain, not something any buffer can fix.
        let (result, errs) = neon_compiler::typecheck::check::check_all(&mut env, &modules);
        let (mut diagnostics, mut foreign) = route(
            errs.iter().map(|e| (e.module.clone(), convert(e))),
            &doc_module,
            &sources,
        );

        // The checker's warnings, routed on the same terms as its errors. A warning
        // never blocks `checked` below, so hover and navigation still come from a
        // warned-but-clean check.
        let (own_warns, foreign_warns) = route(
            result.warnings.iter().map(|w| {
                (
                    w.module.clone(),
                    Diagnostic {
                        span: w.span.clone(),
                        message: w.message.clone(),
                        labels: Vec::new(),
                        help: None,
                        severity: Severity::Warning,
                        code: Some(w.lint.name()),
                        fix: None,
                    },
                )
            }),
            &doc_module,
            &sources,
        );
        diagnostics.extend(own_warns);
        // `route` returns one entry per source, in source order, so the two zip.
        for ((_, ds), (_, extra)) in foreign.iter_mut().zip(foreign_warns) {
            ds.extend(extra);
        }

        // `modules` borrows `module`, and `Checked` owns it — so the borrow has to end
        // before the move. Nothing above needs it past this point.
        drop(modules);
        let foreign = with_text(foreign, &sources);
        Analysis {
            diagnostics,
            foreign,
            checked: Some(Checked {
                module,
                result,
                env,
                sources,
            }),
        }
    }
}

/// Lex, parse and expand a sibling project file, with no diagnostics: its own editor
/// window is where its errors belong, and this run only wants its declarations.
fn parse_quietly(text: &str, config: &expand::Config) -> Option<ast::Module> {
    let tokens = lexer::lex(text).ok()?;
    let (module, errors) = parser::parse(&tokens, text.len());
    if !errors.is_empty() {
        return None;
    }
    let (expanded, _meta, expand_errors) = expand::expand(module?, config);
    if !expand_errors.is_empty() {
        return None;
    }
    Some(expanded)
}

/// Split diagnostics between the analyzed document and the project files they belong to.
/// Every project file gets an entry, EMPTY when it is clean — publishing nothing is how
/// a fixed error leaves the problems panel. A stdlib module's match no source here and
/// are dropped: anchored to any buffer they would underline an unrelated token, and a
/// broken stdlib is a broken toolchain.
fn route(
    items: impl Iterator<Item = (Vec<String>, Diagnostic)>,
    doc_module: &[String],
    sources: &[Source],
) -> (Vec<Diagnostic>, Vec<(PathBuf, Vec<Diagnostic>)>) {
    let mut own = Vec::new();
    let mut foreign: Vec<(PathBuf, Vec<Diagnostic>)> = sources
        .iter()
        .map(|s| (s.path.clone(), Vec::new()))
        .collect();
    for (module, d) in items {
        // Exact file first; then the longest non-empty prefix, because an error inside
        // `util`'s `internal mod raw` carries `["util","raw"]` and the file that holds
        // it is `util.neon`. The entry's empty path never matches as a prefix — it
        // would claim everything.
        if module == doc_module {
            own.push(d);
            continue;
        }
        if let Some(i) = sources.iter().position(|s| s.module == module) {
            foreign[i].1.push(d);
            continue;
        }
        let doc_len =
            (!doc_module.is_empty() && module.starts_with(doc_module)).then_some(doc_module.len());
        let best = sources
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.module.is_empty() && module.starts_with(&s.module))
            .max_by_key(|(_, s)| s.module.len());
        match (doc_len, best) {
            (Some(dl), Some((i, s))) if s.module.len() > dl => foreign[i].1.push(d),
            (Some(_), _) => own.push(d),
            (None, Some((i, _))) => foreign[i].1.push(d),
            // A stdlib module's: dropped, not mis-attached.
            (None, None) => {}
        }
    }
    (own, foreign)
}

/// Attach each foreign file's text to its diagnostics; `route` returns one entry per
/// source in source order, which is what makes the zip sound.
fn with_text(
    foreign: Vec<(PathBuf, Vec<Diagnostic>)>,
    sources: &[Source],
) -> Vec<(PathBuf, String, Vec<Diagnostic>)> {
    foreign
        .into_iter()
        .zip(sources)
        .map(|((path, ds), s)| (path, s.index.text().to_string(), ds))
        .collect()
}

/// The active `@cfg` keys are this machine's, matching what a build here would do. An
/// editor showing code as live that the local target would drop is worse than the reverse,
/// since the dropped branch is the one nobody is checking.
fn config() -> expand::Config {
    expand::Config::with([
        std::env::consts::OS.to_string(),
        std::env::consts::ARCH.to_string(),
    ])
}

/// A checker error, with the labels and help it carries. Both are dropped by the plain
/// path above because lexer and parser errors have neither.
fn convert(e: &neon_compiler::typecheck::env::TypeError) -> Diagnostic {
    use neon_compiler::typecheck::env::{Suggestion, TypeErrorKind};
    // The one mechanical fix the checker computes today: the missing import behind an
    // unknown name. DidYouMean stays a help line — replacing what the user typed is a
    // judgement, adding an import is not.
    let fix = match &e.kind {
        TypeErrorKind::UnknownName {
            suggestion: Some(Suggestion::AddUse { path, .. }),
            ..
        }
        | TypeErrorKind::Unknown {
            suggestion: Some(Suggestion::AddUse { path, .. }),
            ..
        } => Some(Fix::InsertUse(path.clone())),
        _ => None,
    };
    Diagnostic {
        span: e.span.clone(),
        message: e.to_string(),
        labels: e.labels(),
        help: e.help(),
        severity: Severity::Error,
        code: None,
        fix,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn analyzer() -> Analyzer {
        let dir = std::env::var_os("NEON_STDLIB")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../neon/stdlib")
            });
        let mut sources = Vec::new();
        collect(&dir, &dir, &mut sources);
        sources.sort();
        assert!(!sources.is_empty(), "no stdlib found at {}", dir.display());
        Analyzer::new(&dir, &sources).expect("the stdlib parses")
    }

    fn collect(root: &std::path::Path, dir: &std::path::Path, out: &mut Vec<(String, String)>) {
        for entry in std::fs::read_dir(dir).expect("readable") {
            let path = entry.expect("readable").path();
            if path.is_dir() {
                collect(root, &path, out);
            } else if path.extension().is_some_and(|e| e == "neon") {
                let rel = path.strip_prefix(root).expect("under root");
                let text = std::fs::read_to_string(&path).expect("readable");
                out.push((rel.to_string_lossy().replace('\\', "/"), text));
            }
        }
    }

    /// A throwaway project on disk; analysis discovers it from the document's path.
    fn project(tag: &str, files: &[(&str, &str)]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("neon_lsp_project_{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src")).expect("mkdir");
        std::fs::write(dir.join("neon.toml"), "[package]\nname = \"app\"\n").expect("manifest");
        for (rel, text) in files {
            let p = dir.join(rel);
            std::fs::create_dir_all(p.parent().expect("parent")).expect("mkdirs");
            std::fs::write(&p, text).expect("write");
        }
        dir
    }

    const ENTRY: &str =
        "use std::io;\nuse util;\n\nfn main() {\n    io::println(util::greet());\n}\n";
    const UTIL: &str = "fn greet() -> str { \"hi\" }\n";

    #[test]
    fn a_module_resolves_its_siblings() {
        let dir = project(
            "resolves",
            &[("src/main.neon", ENTRY), ("src/util.neon", UTIL)],
        );
        let a = analyzer();

        // The entry sees the module...
        let entry = a.analyze(Some(&dir.join("src/main.neon")), ENTRY, &[]);
        assert!(
            entry.diagnostics.is_empty(),
            "{:?}",
            entry
                .diagnostics
                .iter()
                .map(|d| &d.message)
                .collect::<Vec<_>>()
        );
        // ...and its check knows which file `util` is, so a jump can open it.
        let checked = entry.checked.expect("clean check");
        let src = checked
            .source_of(&a, &["util".to_string()])
            .expect("util has a file");
        assert!(src.path.ends_with("util.neon"));

        // The module, analyzed as the open document, sees the entry the same way.
        let util = a.analyze(Some(&dir.join("src/util.neon")), UTIL, &[]);
        assert!(util.diagnostics.is_empty());
    }

    #[test]
    fn a_siblings_error_is_routed_to_its_file_not_mine() {
        // The signature stays `str`, so the entry's call remains well-typed and the
        // one error in the program is inside util's body.
        let broken = "fn greet() -> str { 41 }\n";
        let dir = project(
            "routes",
            &[("src/main.neon", ENTRY), ("src/util.neon", broken)],
        );
        let a = analyzer();

        let entry = a.analyze(Some(&dir.join("src/main.neon")), ENTRY, &[]);
        assert!(
            entry.diagnostics.is_empty(),
            "the entry did nothing wrong: {:?}",
            entry
                .diagnostics
                .iter()
                .map(|d| &d.message)
                .collect::<Vec<_>>()
        );
        let (path, _, ds) = entry
            .foreign
            .iter()
            .find(|(p, _, _)| p.ends_with("util.neon"))
            .expect("util.neon has an entry");
        assert!(path.ends_with("util.neon"));
        assert_eq!(ds.len(), 1, "{ds:?}");
    }

    #[test]
    fn an_open_buffer_overlays_its_disk_file() {
        let broken = "fn greet() -> str { 41 }\n";
        let dir = project(
            "overlays",
            &[("src/main.neon", ENTRY), ("src/util.neon", broken)],
        );
        let a = analyzer();

        // util.neon is broken on disk, but the editor's buffer has fixed it: the
        // analysis of main must see the buffer, not the file.
        let util_path = std::fs::canonicalize(dir.join("src/util.neon")).expect("exists");
        let overlays = vec![(util_path, UTIL.to_string())];
        let entry = a.analyze(Some(&dir.join("src/main.neon")), ENTRY, &overlays);
        assert!(entry.diagnostics.is_empty());
        let (_, _, ds) = entry
            .foreign
            .iter()
            .find(|(p, _, _)| p.ends_with("util.neon"))
            .expect("util.neon has an entry");
        assert!(ds.is_empty(), "the overlay fixed it: {ds:?}");
    }

    #[test]
    fn a_lone_file_is_not_a_project() {
        let a = analyzer();
        let lone = "use std::io;\nfn main() { io::println(\"hi\"); }\n";
        let out = a.analyze(None, lone, &[]);
        assert!(out.diagnostics.is_empty());
        assert!(out.foreign.is_empty());
        assert!(out.checked.expect("clean").sources.is_empty());
    }
}
