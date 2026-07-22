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
pub struct Diagnostic {
    pub span: Range<usize>,
    pub message: String,
    /// Extra spans that explain the error, each with its own note. Rendered as related
    /// information so an editor can offer to jump to them.
    pub labels: Vec<(Range<usize>, String)>,
    pub help: Option<String>,
}

impl Diagnostic {
    fn plain(span: Range<usize>, message: String) -> Self {
        Diagnostic { span, message, labels: Vec::new(), help: None }
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
    /// The user's module, expanded and numbered — the same AST the checker saw, so an
    /// `ExprId` from `result` indexes into it.
    pub module: ast::Module,
    pub result: TypecheckResult,
    pub env: Env,
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
    pub diagnostics: Vec<Diagnostic>,
    pub checked: Option<Checked>,
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

        Ok(Analyzer { cached: Some(Cached { modules, next_id, sources }), config: config() })
    }

    /// Lexer and parser diagnostics only, for a session with no usable toolchain.
    ///
    /// The checker is skipped rather than run against an empty stdlib on purpose: with no
    /// `std` in scope every single name in a normal file is undefined, and the resulting
    /// wall of red says nothing true about the user's code.
    pub fn syntax_only() -> Analyzer {
        Analyzer { cached: None, config: config() }
    }

    /// The stdlib files, for turning a `DefSite` into a location.
    pub fn sources(&self) -> &[Source] {
        self.cached.as_ref().map(|c| c.sources.as_slice()).unwrap_or_default()
    }

    /// The stdlib file a module was declared in.
    pub fn source_of(&self, module: &[String]) -> Option<&Source> {
        self.sources().iter().find(|s| s.module == module)
    }

    /// Everything the front end can say about one file, in pipeline order.
    pub fn analyze(&self, src: &str) -> Analysis {
        let bail = |d: Vec<Diagnostic>| Analysis { diagnostics: d, checked: None };

        let tokens = match lexer::lex(src) {
            Ok(t) => t,
            Err(errors) => {
                return bail(
                    errors.iter().map(|e| Diagnostic::plain(e.span.clone(), e.to_string())).collect(),
                )
            }
        };

        let (module, errors) = parser::parse(&tokens, src.len());
        if !errors.is_empty() {
            return bail(
                errors.iter().map(|e| Diagnostic::plain(e.span.clone(), e.to_string())).collect(),
            );
        }
        let Some(mut module) = module else { return bail(Vec::new()) };

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

        let Some(cached) = &self.cached else { return bail(Vec::new()) };

        // Numbering the stdlib first and the user's module after it keeps every `ExprId`
        // in the compilation unique, which is what lets one `TypecheckResult` cover both.
        ast::number_exprs_from(&mut module, cached.next_id);

        let mut modules: Vec<(Vec<String>, &ast::Module)> =
            cached.modules.iter().map(|(p, m)| (p.clone(), m)).collect();
        modules.push((Vec::new(), &module));

        // `RootApplication` because an editor is nearly always looking at a program: it is
        // the stricter of the two (a library has no `main` to demand), so this errs toward
        // showing a diagnostic rather than hiding one.
        let mut env = Env::build_with(&modules, Unit::RootApplication);
        if !env.errors().is_empty() {
            return bail(env.errors().iter().map(convert).collect());
        }

        // Every diagnostic of the run, resolution errors included: `check_all` drains the
        // environment's channel into what it returns, so an unknown type written inside a
        // body reaches the editor rather than vanishing.
        let (result, errs) = neon_compiler::typecheck::check::check_all(&mut env, &modules);
        let diagnostics = errs.iter().map(convert).collect();

        // `modules` borrows `module`, and `Checked` owns it — so the borrow has to end
        // before the move. Nothing above needs it past this point.
        drop(modules);
        Analysis { diagnostics, checked: Some(Checked { module, result, env }) }
    }
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
    Diagnostic {
        span: e.span.clone(),
        message: e.to_string(),
        labels: e.labels(),
        help: e.help(),
    }
}
