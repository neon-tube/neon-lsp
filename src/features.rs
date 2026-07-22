//! Answering questions about a checked file.
//!
//! Everything here is a query over an `analysis::Checked` — the module's AST, the
//! `TypecheckResult` the checker produced, and the `Env` that can print a `TyId`. None of
//! it re-runs the front end, and none of it re-derives anything the checker already
//! decided. That is the rule the whole module is built on, and it is the same rule
//! `typecheck/result.rs` states for lowering: **ask, never derive.** A hover that
//! recomputed a type would be a second type checker, and the two would disagree.
//!
//! The shared shape of a position-taking request is: turn the LSP position into a byte
//! offset (`LineIndex::offset`), find the innermost AST node containing it
//! (`ast::visit::innermost_at`), and look its `ExprId` up in one of the checker's maps.
//! Three steps, no index to keep warm, and the only cost is one walk of the module.

use crate::analysis::{Analyzer, Checked, Source};
use crate::position::LineIndex;
use lsp_types::{
    CompletionItem, CompletionItemKind, Documentation, Location, MarkupContent, MarkupKind,
    ParameterInformation, ParameterLabel, Position, Range, SemanticToken, SemanticTokenType,
    SignatureHelp, SignatureInformation, SymbolKind, Uri,
};
use neon_compiler::ast::{self, Expr, ExprId};
use neon_compiler::lexer::{Span, TriviaKind};
use neon_compiler::typecheck::env::FnSig;
use neon_compiler::typecheck::print;
use neon_compiler::typecheck::result::{DefKind, DefSite};
use neon_compiler::typecheck::{Env, TyId};

/// A span turned into an LSP range within one file.
fn range(index: &LineIndex, span: &Span) -> Range {
    Range { start: index.position(span.start), end: index.position(span.end) }
}

/// `ty` as Neon type syntax.
///
/// Takes the whole `Env` because printing interns the complement of a negated type, which
/// needs the type table mutably. Nothing observable is added — the table is hash-consed —
/// so this is a read as far as any caller is concerned.
fn show(env: &mut Env, ty: TyId) -> String {
    print::print(&mut env.solver.t, ty)
}

// ---- documentation ----

/// The `///` block immediately above `span`, as Markdown.
///
/// Doc comments never reach the AST: the lexer records them as trivia and the parser,
/// which has no field to put them in, drops them. So the text is recovered here by going
/// back to the trivia table and taking the run of `Doc` comments that ends where the
/// declaration begins.
///
/// "Immediately above" is the whole subtlety. A run is only the declaration's own
/// documentation if nothing but whitespace separates the two — otherwise the `///` block
/// documenting the *previous* function would be shown for this one, which is worse than
/// showing nothing because it is confidently wrong. So the walk stops at the first gap
/// containing a blank line.
pub fn doc_above(text: &str, trivia: &[neon_compiler::lexer::Trivia], span: &Span) -> Option<String> {
    let mut lines: Vec<&str> = Vec::new();
    let mut boundary = span.start;

    for t in trivia.iter().rev() {
        if t.span.end > boundary || t.kind != TriviaKind::Doc {
            // Trivia is in source order, so the first one that is not an adjacent doc
            // comment ends the run — but only once we are actually above the span.
            if t.span.end > boundary {
                continue;
            }
            break;
        }
        // Whatever sits between this comment and what follows it. A blank line here
        // means the comment belongs to something else.
        let gap = &text[t.span.end..boundary];
        if !gap.trim().is_empty() || gap.matches('\n').count() > 1 {
            break;
        }
        lines.push(t.text.trim());
        boundary = t.span.start;
    }

    if lines.is_empty() {
        return None;
    }
    lines.reverse();
    Some(lines.join("\n"))
}

/// The documentation for a definition, wherever it lives.
///
/// A user's own function is documented in the document being edited; a stdlib function is
/// documented in a file the server parsed at startup. Both are answered here so no caller
/// has to know which kind it is holding.
fn doc_for(analyzer: &Analyzer, current: &LineIndex, site: &DefSite) -> Option<String> {
    match analyzer.source_of(&site.module) {
        Some(src) => doc_above(src.index.text(), &src.trivia, &site.span),
        // The module is not a stdlib one, so it is the user's own file. Its trivia is not
        // kept — it is re-lexed here rather than on every keystroke, since hovering is
        // rare next to typing and lexing one file is cheap.
        None => {
            let lexed = neon_compiler::lexer::lex_full(current.text()).ok()?;
            doc_above(current.text(), &lexed.trivia, &site.span)
        }
    }
}

// ---- signatures ----

/// A function signature as Neon source, the way its declaration reads.
///
/// Reconstructed from the `FnSig` rather than sliced out of the file, because the two are
/// not the same text: a generic signature's parameters are printed *solved* at a call
/// site, which is the useful thing to show. `fn push(xs: List[i64], x: i64)` tells the
/// reader what this call does; `fn push[T](xs: List[T], x: T)` makes them work it out.
pub fn signature(env: &mut Env, sig: &FnSig) -> String {
    let generics = if sig.generics.is_empty() {
        String::new()
    } else {
        format!("[{}]", sig.generics.join(", "))
    };
    let params = sig
        .params
        .iter()
        .map(|(n, t)| format!("{n}: {}", show(env, *t)))
        .collect::<Vec<_>>()
        .join(", ");

    // `never` is how "no `throws` clause" is spelled internally; printing it would put a
    // clause on every non-throwing function in the stdlib.
    let never = env.solver.t.never();
    let throws =
        if sig.throws == never { String::new() } else { format!(" throws {}", show(env, sig.throws)) };

    let wheres = if sig.wheres.is_empty() {
        String::new()
    } else {
        let bounds = sig
            .wheres
            .iter()
            .map(|(v, p)| format!("{v}: {}", p.join("::")))
            .collect::<Vec<_>>()
            .join(", ");
        format!(" where {bounds}")
    };

    let ret = show(env, sig.ret);
    format!("fn {}{generics}({params}){throws} -> {ret}{wheres}", sig.name)
}

/// Fence a piece of Neon source for display in an editor's hover popup.
fn fenced(code: &str) -> String {
    format!("```neon\n{code}\n```")
}

// ---- hover ----

/// What to show for the thing under the cursor.
///
/// The answer is always the same three parts, in the same order, because a hover that
/// varies its layout by what it found is one the eye cannot skim: the thing's type or
/// signature as fenced Neon, then its documentation, then where it came from. Any part
/// that has no content is omitted rather than rendered empty.
pub fn hover(
    analyzer: &Analyzer,
    checked: &mut Checked,
    index: &LineIndex,
    pos: Position,
) -> Option<(MarkupContent, Range)> {
    let offset = index.offset(pos);

    // Two things can be under the cursor. A *use* of a name is an expression the checker
    // typed; a *binding* is a `Pattern`, which it did not — patterns carry no entry in
    // `expr_types`. Both must answer, because hovering the `x` in `let x = ...` to ask
    // what it is is at least as common as hovering a use of it.
    let (id, span, site) = match ast::visit::innermost_at(&checked.module, offset) {
        Some(expr) => (Some(expr.id), expr.span.clone(), checked.result.def(expr.id).cloned()),
        None => {
            let site = binding_at(checked, offset)?;
            let id = type_carrier(checked, &site);
            (id, site.span.clone(), Some(site))
        }
    };

    let mut parts: Vec<String> = Vec::new();

    // A name that resolved to a function shows its signature; anything else shows the
    // type the checker gave the expression. Both come out of the same maps.
    let sig = site
        .as_ref()
        .filter(|d| d.kind == DefKind::Fn)
        .and_then(|d| fn_at(&checked.env, d).cloned());

    match sig {
        Some(sig) => parts.push(fenced(&signature(&mut checked.env, &sig))),
        None => {
            let ty = checked.result.ty(id?)?;
            parts.push(fenced(&show(&mut checked.env, ty)));
        }
    }

    if let Some(site) = site {
        if let Some(doc) = doc_for(analyzer, index, &site) {
            parts.push(doc);
        }
        // Where it came from, but only when that is not "right here" — labelling every
        // local with the module it is obviously in is noise.
        if !site.module.is_empty() {
            parts.push(format!("*in* `{}`", site.module.join("::")));
        }
    }

    let content = MarkupContent {
        kind: MarkupKind::Markdown,
        value: parts.join("\n\n---\n\n"),
    };
    Some((content, range(index, &span)))
}

/// The binding whose own span contains `offset` — the cursor on a declaration rather
/// than a use.
fn binding_at(checked: &Checked, offset: usize) -> Option<DefSite> {
    checked.result.defs().map(|(_, d)| d).find(|d| d.span.contains(&offset)).cloned()
}

/// An expression whose type is the type of `site`.
///
/// A binding has no type of its own recorded — only expressions do — so its type is read
/// off something that does have one. A `let`'s initialiser is the exact answer and is
/// tried first. Failing that (a parameter has no initialiser), any *use* of the binding
/// will do: a use resolves to this same site and the checker typed it. A binding that is
/// never used and is not a `let` has no answer, which is correct rather than a gap —
/// there is nothing in the program that fixes its type.
fn type_carrier(checked: &Checked, site: &DefSite) -> Option<ExprId> {
    let mut found = None;
    let mut v = LetOf { want: &site.span, found: &mut found };
    for d in &checked.module.decls {
        ast::visit::walk_decl(&mut v, d);
    }
    if found.is_some() {
        return found;
    }
    checked.result.defs().find(|(_, d)| *d == site).map(|(e, _)| e)
}

/// Finds the initialiser of the `let` that binds a given span.
struct LetOf<'a> {
    want: &'a Span,
    found: &'a mut Option<ExprId>,
}

impl<'a> ast::visit::Visitor<'a> for LetOf<'_> {
    fn stmt(&mut self, s: &'a ast::Stmt) {
        if let ast::StmtKind::Let { pat, value, .. } = &s.kind {
            if pat.span == *self.want {
                *self.found = Some(value.id);
            }
        }
        ast::visit::walk_stmt(self, s);
    }
}

/// The signature a `DefSite` names, if it names a function.
///
/// Matched on span rather than name: two functions in different modules share a name
/// often, and the span is what the checker actually recorded.
fn fn_at<'e>(env: &'e Env, site: &DefSite) -> Option<&'e FnSig> {
    env.fns().iter().find(|f| f.span == site.span && f.module == site.module)
}

// ---- go to definition ----

/// Where the name under the cursor was defined.
pub fn definition(
    analyzer: &Analyzer,
    checked: &Checked,
    index: &LineIndex,
    uri: &Uri,
    pos: Position,
) -> Option<Location> {
    let offset = index.offset(pos);
    let expr = ast::visit::innermost_at(&checked.module, offset)?;
    let site = checked.result.def(expr.id)?;
    locate(analyzer, index, uri, site)
}

/// A `DefSite` as a location an editor can open.
///
/// The module decides which file, and the test is "is this one of the stdlib's" rather
/// than "is the module path empty". The empty path belongs to the document's *root*
/// module, but a declaration inside `mod inner { .. }` is recorded against `["inner"]` —
/// the checker names the module the code was written in, not the file it came from. Keying
/// off emptiness therefore lost every name declared inside a `mod` block: `source_of`
/// found no stdlib file called `inner`, and go-to-definition, find-references and rename
/// all silently returned nothing inside one.
fn locate(analyzer: &Analyzer, index: &LineIndex, uri: &Uri, site: &DefSite) -> Option<Location> {
    let Some(src) = analyzer.source_of(&site.module) else {
        return Some(Location { uri: uri.clone(), range: range(index, &site.span) });
    };
    let src: &Source = src;
    // Through `url` and back out as a string: it is the only one of the two that knows
    // how to percent-encode a filesystem path, and `Uri` is the only one the protocol
    // accepts.
    let target = url::Url::from_file_path(&src.path).ok()?;
    let target: Uri = target.as_str().parse().ok()?;
    Some(Location { uri: target, range: range(&src.index, &site.span) })
}

// ---- find references, rename ----

/// Every occurrence in this document of whatever the cursor is on.
///
/// This is `resolved_names` read backwards: two names refer to the same thing exactly
/// when the checker recorded the same `DefSite` for both. That is why it gets shadowing
/// right for free — the inner `x` and the outer `x` have different sites, so they are
/// different answers, and no amount of matching on the text could have told them apart.
///
/// The definition itself is included when it is in this file, since an editor listing
/// references is expected to show it.
pub fn references(
    analyzer: &Analyzer,
    checked: &Checked,
    index: &LineIndex,
    pos: Position,
) -> Vec<Range> {
    let Some(target) = site_under(checked, index, pos) else { return Vec::new() };

    let mut spans: Vec<Span> = spans_of(checked, &target);
    // The definition too, when it is in this file — see `locate` for why the test is
    // "not a stdlib module" rather than "no module".
    if analyzer_has_no_source(analyzer, &target) {
        spans.push(target.span.clone());
    }

    spans.sort_by_key(|s| (s.start, s.end));
    spans.dedup();
    spans.iter().map(|s| range(index, s)).collect()
}

/// Whether a definition lives in the document being edited rather than the stdlib.
///
/// Phrased as "the analyzer has no file for this module", because that is the only thing
/// that reliably distinguishes the two: the stdlib's modules are exactly the ones parsed
/// from files at startup, and everything else — root or nested `mod` alike — is the user's.
fn analyzer_has_no_source(analyzer: &Analyzer, site: &DefSite) -> bool {
    analyzer.source_of(&site.module).is_none()
}

/// The spans in this module of every name resolving to `target`.
fn spans_of(checked: &Checked, target: &DefSite) -> Vec<Span> {
    let mut ids: Vec<ExprId> = checked
        .result
        .defs()
        .filter(|(_, d)| *d == target)
        .map(|(e, _)| e)
        .collect();
    ids.sort_by_key(|e| e.0);

    let mut spans = Vec::new();
    ast::visit::each_expr(&checked.module, |e: &Expr| {
        if ids.binary_search_by_key(&e.id.0, |i| i.0).is_ok() {
            spans.push(e.span.clone());
        }
    });
    spans
}

/// The definition the cursor is on — whether it is sitting on a use of the name or on the
/// binding itself.
///
/// Both have to work: a user renames by putting the cursor on the declaration at least as
/// often as on a use. A use is a resolved expression; a binding is a `Pattern`, which
/// resolves to nothing, so it is found by looking for a site whose span is the one the
/// cursor is inside.
fn site_under(checked: &Checked, index: &LineIndex, pos: Position) -> Option<DefSite> {
    let offset = index.offset(pos);

    if let Some(e) = ast::visit::innermost_at(&checked.module, offset) {
        if let Some(d) = checked.result.def(e.id) {
            return Some(d.clone());
        }
    }
    // On the binding site itself.
    checked
        .result
        .defs()
        .map(|(_, d)| d)
        .find(|d| d.span.contains(&offset))
        .cloned()
}

/// The edits that rename whatever the cursor is on.
///
/// Refuses anything defined outside this document. Renaming into the stdlib would mean
/// editing the user's toolchain — files they did not open and probably cannot write — and
/// a rename that silently skipped those files instead would leave the project not
/// compiling. Declining is the only safe answer.
pub fn rename(
    analyzer: &Analyzer,
    checked: &Checked,
    index: &LineIndex,
    pos: Position,
) -> Option<Vec<Range>> {
    let target = site_under(checked, index, pos)?;
    if !analyzer_has_no_source(analyzer, &target) {
        return None;
    }
    let mut spans = spans_of(checked, &target);
    spans.push(target.span.clone());
    spans.sort_by_key(|s| (s.start, s.end));
    spans.dedup();
    Some(spans.iter().map(|s| range(index, s)).collect())
}

// ---- document symbols ----

/// One entry in the outline, with whatever it contains.
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    pub range: Range,
    pub children: Vec<Symbol>,
}

/// The outline of one file: every declaration, in source order.
///
/// Read off the AST rather than the `Env`, because the outline has to match the file the
/// user is looking at. `Env::fns` holds the whole compilation, stdlib included.
///
/// Nested rather than flat. A `mod` containing ten functions is the case the outline is
/// for, and flattening it produces ten entries with no indication of what they belong to.
pub fn document_symbols(checked: &Checked, index: &LineIndex) -> Vec<Symbol> {
    let mut out = Vec::new();
    collect_symbols(&checked.module.decls, index, &mut out);
    out
}

fn collect_symbols(decls: &[ast::Decl], index: &LineIndex, out: &mut Vec<Symbol>) {
    for d in decls {
        let entry = match &d.kind {
            ast::DeclKind::Fn(f) => Some((f.name.clone(), SymbolKind::FUNCTION)),
            ast::DeclKind::Record(r) => Some((r.name.clone(), SymbolKind::STRUCT)),
            ast::DeclKind::Protocol(p) => Some((p.name.clone(), SymbolKind::INTERFACE)),
            ast::DeclKind::TypeAlias(a) => Some((a.name.clone(), SymbolKind::TYPE_PARAMETER)),
            ast::DeclKind::MuType(a) => Some((a.name.clone(), SymbolKind::TYPE_PARAMETER)),
            ast::DeclKind::Newtype(a) => Some((a.name.clone(), SymbolKind::TYPE_PARAMETER)),
            ast::DeclKind::Const(c) => Some((c.name.clone(), SymbolKind::CONSTANT)),
            ast::DeclKind::Mod(m) => {
                let mut children = Vec::new();
                collect_symbols(&m.decls, index, &mut children);
                out.push(Symbol {
                    name: m.name.clone(),
                    kind: SymbolKind::MODULE,
                    range: range(index, &d.span),
                    children,
                });
                None
            }
            // An `impl` is named by what it implements, which is a type rather than a
            // name, so it is shown as its own group with the methods inside.
            ast::DeclKind::Impl(i) => {
                let children = i
                    .methods
                    .iter()
                    .map(|m| Symbol {
                        name: m.name.clone(),
                        kind: SymbolKind::METHOD,
                        range: range(index, &d.span),
                        children: Vec::new(),
                    })
                    .collect();
                out.push(Symbol {
                    name: format!("impl {}", i.protocol.join("::")),
                    kind: SymbolKind::CLASS,
                    range: range(index, &d.span),
                    children,
                });
                None
            }
            ast::DeclKind::Use(_) | ast::DeclKind::TestBlock(_) => None,
            ast::DeclKind::Error => None,
        };
        if let Some((name, kind)) = entry {
            out.push(Symbol {
                name,
                kind,
                range: range(index, &d.span),
                children: Vec::new(),
            });
        }
    }
}

// ---- completion ----

/// What could go where the cursor is.
///
/// Every function the current module can see, with its signature as the detail line and
/// its `///` block as the documentation. This is the list `Env` already holds — the
/// completion is a projection of the same table name resolution consults, so anything
/// offered here is a name that would actually resolve.
///
/// Locals are added from the checker's record of what is bound, filtered to the bindings
/// whose scope the cursor is actually inside. Without that filter a completion list in one
/// function offers the locals of every other function in the file.
pub fn completions(
    analyzer: &Analyzer,
    checked: &mut Checked,
    index: &LineIndex,
    pos: Position,
) -> Vec<CompletionItem> {
    let offset = index.offset(pos);
    let mut out = Vec::new();

    // Locals first: they shadow functions, and an editor preserves the order it is given
    // for items of equal score.
    for (name, ty, site) in locals_in_scope(checked, index.text(), offset) {
        out.push(CompletionItem {
            label: name,
            kind: Some(match site {
                DefKind::Param => CompletionItemKind::VARIABLE,
                _ => CompletionItemKind::VARIABLE,
            }),
            detail: Some(show(&mut checked.env, ty)),
            ..Default::default()
        });
    }

    let sigs: Vec<FnSig> = checked.env.fns().to_vec();
    for sig in sigs {
        // A protocol's required method has no body and is not callable by that name; it
        // is reached through dispatch on an implementing type.
        if !sig.has_body {
            continue;
        }
        let detail = signature(&mut checked.env, &sig);
        let documentation = analyzer
            .source_of(&sig.module)
            .and_then(|src| doc_above(src.index.text(), &src.trivia, &sig.span))
            .map(|value| Documentation::MarkupContent(MarkupContent {
                kind: MarkupKind::Markdown,
                value,
            }));

        out.push(CompletionItem {
            label: sig.name.clone(),
            kind: Some(CompletionItemKind::FUNCTION),
            detail: Some(detail),
            documentation,
            // Qualified names sort after bare ones so the prelude, which is what people
            // mean most of the time, is not buried under `std::`.
            sort_text: Some(format!("{}{}", if sig.module.is_empty() { "0" } else { "1" }, sig.name)),
            ..Default::default()
        });
    }

    out
}

/// The locals visible at `offset`, innermost binding first.
///
/// Scope is approximated by span containment: a binding is in scope at a point if the
/// function that binds it contains that point and the binding comes before it. That is
/// not the checker's own scope rule — it does not model a block ending — so it can offer
/// a name from a sibling block that has closed. Erring this way is deliberate: an extra
/// candidate in a completion list costs a keystroke, and a missing one costs the feature.
fn locals_in_scope(checked: &Checked, text: &str, offset: usize) -> Vec<(String, TyId, DefKind)> {
    let enclosing = checked
        .module
        .decls
        .iter()
        .find(|d| d.span.contains(&offset))
        .map(|d| d.span.clone());
    let Some(enclosing) = enclosing else { return Vec::new() };

    let mut seen: Vec<(String, TyId, DefKind)> = Vec::new();
    for (id, site) in checked.result.defs() {
        if site.kind == DefKind::Fn || !enclosing.contains(&site.span.start) {
            continue;
        }
        if site.span.start > offset {
            continue;
        }
        let Some(ty) = checked.result.ty(id) else { continue };
        // A binding's span is exactly its name, so the source text is the lookup — the
        // AST stores the name on the `Pattern`, which a `DefSite` cannot reach.
        let Some(name) = text.get(site.span.clone()) else { continue };
        if !seen.iter().any(|(n, ..)| n == name) {
            seen.push((name.to_string(), ty, site.kind));
        }
    }
    seen
}

// ---- signature help ----

/// The signature of the call the cursor is inside, and which argument it is on.
pub fn signature_help(
    checked: &mut Checked,
    index: &LineIndex,
    pos: Position,
) -> Option<SignatureHelp> {
    let offset = index.offset(pos);

    // The innermost *call* containing the cursor, which is not the innermost expression:
    // the cursor is on an argument, and the argument is not the thing being described.
    let mut best: Option<(&Expr, &Expr, usize)> = None;
    ast::visit::each_expr(&checked.module, |e| {
        let ast::ExprKind::Call { callee, args, .. } = &e.kind else { return };
        if !e.span.contains(&offset) {
            return;
        }
        let active = args.iter().position(|a| a.span.contains(&offset)).unwrap_or(args.len().saturating_sub(1));
        let width = e.span.end - e.span.start;
        if best.is_none_or(|(b, ..)| width <= b.span.end - b.span.start) {
            best = Some((e, callee, active));
        }
    });
    let (_, callee, active) = best?;

    let site = checked.result.def(callee.id)?.clone();
    let sig = fn_at(&checked.env, &site)?.clone();

    let params: Vec<ParameterInformation> = sig
        .params
        .iter()
        .map(|(n, t)| ParameterInformation {
            label: ParameterLabel::Simple(format!("{n}: {}", show(&mut checked.env, *t))),
            documentation: None,
        })
        .collect();

    Some(SignatureHelp {
        signatures: vec![SignatureInformation {
            label: signature(&mut checked.env, &sig),
            documentation: None,
            parameters: Some(params),
            active_parameter: Some(active as u32),
        }],
        active_signature: Some(0),
        active_parameter: Some(active as u32),
    })
}

// ---- inlay hints ----

/// Type annotations for bindings that did not write one.
///
/// Only for `let` without an annotation: a hint restating what the user typed is noise,
/// and the whole value of the feature is showing what inference concluded.
pub fn inlay_hints(checked: &mut Checked, index: &LineIndex) -> Vec<(Position, String)> {
    let mut out = Vec::new();
    let mut pending: Vec<(Span, ExprId)> = Vec::new();

    collect_lets(&checked.module.decls, &mut pending);
    for (span, value) in pending {
        let Some(ty) = checked.result.ty(value) else { continue };
        out.push((index.position(span.end), format!(": {}", show(&mut checked.env, ty))));
    }
    out
}

fn collect_lets(decls: &[ast::Decl], out: &mut Vec<(Span, ExprId)>) {
    struct V<'o>(&'o mut Vec<(Span, ExprId)>);
    impl<'a> ast::visit::Visitor<'a> for V<'_> {
        fn stmt(&mut self, s: &'a ast::Stmt) {
            if let ast::StmtKind::Let { pat, value, ty: None, .. } = &s.kind {
                if matches!(pat.kind, ast::PatternKind::Bind(_)) {
                    self.0.push((pat.span.clone(), value.id));
                }
            }
            ast::visit::walk_stmt(self, s);
        }
    }
    let mut v = V(out);
    for d in decls {
        ast::visit::walk_decl(&mut v, d);
    }
}

// ---- semantic tokens ----

/// The token types this server emits, in the order the protocol indexes them.
///
/// Deliberately short. Semantic tokens are a *layer over* syntax highlighting, not a
/// replacement for it: tree-sitter and TextMate already colour keywords, strings and
/// numbers from the text alone, and re-sending those over the wire on every edit would be
/// bytes spent to change nothing. What they cannot do is tell a parameter from a local
/// from a function that happens to be used as a value, because that needs name resolution.
/// So this emits exactly the categories resolution knows and syntax does not.
pub const TOKEN_TYPES: &[SemanticTokenType] = &[
    SemanticTokenType::FUNCTION,
    SemanticTokenType::PARAMETER,
    SemanticTokenType::VARIABLE,
];

const TOK_FUNCTION: u32 = 0;
const TOK_PARAMETER: u32 = 1;
const TOK_VARIABLE: u32 = 2;

/// Resolution-aware colouring for every name in the file.
///
/// The protocol wants tokens delta-encoded against each other and in strictly increasing
/// position order, so the walk collects absolute positions first and sorts. Getting the
/// order wrong does not error — the client silently renders garbage from the first
/// out-of-order token on, which is why the sort is unconditional rather than an assertion
/// that the walk already produced them in order.
pub fn semantic_tokens(checked: &Checked, index: &LineIndex) -> Vec<SemanticToken> {
    let mut found: Vec<(Span, u32)> = Vec::new();

    ast::visit::each_expr(&checked.module, |e| {
        // Only names carry resolution. Everything else the syntax layer already knows.
        if !matches!(e.kind, ast::ExprKind::Path(_)) {
            return;
        }
        let Some(site) = checked.result.def(e.id) else { return };
        let kind = match site.kind {
            DefKind::Fn => TOK_FUNCTION,
            DefKind::Param => TOK_PARAMETER,
            // A `const` highlights as a variable. LSP has no `constant` token type -- the
            // convention is `variable` plus a `readonly` modifier, and this server does not
            // send modifiers, so widening the legend would buy nothing a client can use.
            DefKind::Local | DefKind::Const => TOK_VARIABLE,
        };
        found.push((e.span.clone(), kind));
    });

    found.sort_by_key(|(s, _)| (s.start, s.end));

    let mut out = Vec::with_capacity(found.len());
    let mut prev = Position { line: 0, character: 0 };
    for (span, token_type) in found {
        let at = index.position(span.start);
        // A span covering a newline would make `length` meaningless, since the protocol
        // measures it within one line. Names never do, but a malformed span would, and
        // dropping it beats shifting every token after it.
        if index.position(span.end).line != at.line {
            continue;
        }
        let delta_line = at.line - prev.line;
        let delta_start = if delta_line == 0 { at.character - prev.character } else { at.character };
        out.push(SemanticToken {
            delta_line,
            delta_start,
            length: index.position(span.end).character - at.character,
            token_type,
            token_modifiers_bitset: 0,
        });
        prev = at;
    }
    out
}

// ---- folding ----

/// Everything worth collapsing: declaration bodies and the blocks inside them.
///
/// Folded by line rather than by character, which is what editors show. A range that
/// begins and ends on the same line is dropped — an editor renders it as a fold marker
/// that does nothing when clicked.
pub fn folding_ranges(checked: &Checked, index: &LineIndex) -> Vec<(u32, u32)> {
    let mut spans: Vec<Span> = Vec::new();

    for d in &checked.module.decls {
        spans.push(d.span.clone());
        let mut v = Blocks(&mut spans);
        ast::visit::walk_decl(&mut v, d);
    }

    let mut out: Vec<(u32, u32)> = spans
        .iter()
        .map(|s| (index.position(s.start).line, index.position(s.end).line))
        .filter(|(a, b)| b > a)
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Collects every block span, for folding.
struct Blocks<'o>(&'o mut Vec<Span>);

impl<'a> ast::visit::Visitor<'a> for Blocks<'_> {
    fn block(&mut self, b: &'a ast::Block) {
        self.0.push(b.span.clone());
        ast::visit::walk_block(self, b);
    }
}

// ---- selection range ----

/// The chain of expressions containing a position, innermost first.
///
/// This is what drives "expand selection": each press widens to the next enclosing node.
/// Built by collecting every containing span and sorting by width, which gives the nesting
/// order without having to track parent links the AST does not store.
pub fn selection_range(checked: &Checked, index: &LineIndex, pos: Position) -> Vec<Range> {
    let offset = index.offset(pos);
    let mut containing: Vec<Span> = Vec::new();

    ast::visit::each_expr(&checked.module, |e| {
        if e.span.contains(&offset) {
            containing.push(e.span.clone());
        }
    });
    // Patterns as well as expressions. Expanding the selection from the `x` in
    // `let x = ...` is the same gesture as expanding it from a use of `x`, and a walk
    // that only saw expressions jumped straight from the cursor to the whole function.
    ast::visit::each_pattern(&checked.module, |p| {
        if p.span.contains(&offset) {
            containing.push(p.span.clone());
        }
    });
    // Blocks, which are neither, and are the step between a statement and its function.
    let mut blocks = Vec::new();
    for d in &checked.module.decls {
        ast::visit::walk_decl(&mut Blocks(&mut blocks), d);
    }
    containing.extend(blocks.into_iter().filter(|b| b.contains(&offset)));
    // The declaration is the outermost step, and is none of the above.
    for d in &checked.module.decls {
        if d.span.contains(&offset) {
            containing.push(d.span.clone());
        }
    }

    containing.sort_by_key(|s| (s.end - s.start, s.start));
    containing.dedup();
    containing.iter().map(|s| range(index, s)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::Analyzer;

    /// An analyzer over the repository's own stdlib.
    ///
    /// The real one rather than a fixture: these features are mostly *about* the stdlib —
    /// hovering `println` and jumping into `std::io` are the cases that were asked for —
    /// and a stub stdlib would test the plumbing while missing everything that makes the
    /// plumbing worth having.
    fn analyzer() -> Analyzer {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../stdlib");
        let mut sources = Vec::new();
        collect(&dir, &dir, &mut sources);
        sources.sort();
        assert!(!sources.is_empty(), "no stdlib found at {}", dir.display());
        Analyzer::new(&dir, &sources).expect("the repository's own stdlib parses")
    }

    fn collect(root: &std::path::Path, dir: &std::path::Path, out: &mut Vec<(String, String)>) {
        for entry in std::fs::read_dir(dir).expect("the stdlib directory is readable") {
            let path = entry.expect("the entry is readable").path();
            if path.is_dir() {
                collect(root, &path, out);
            } else if path.extension().is_some_and(|e| e == "neon") {
                let rel = path.strip_prefix(root).expect("it is under the root");
                let text = std::fs::read_to_string(&path).expect("the file is readable");
                out.push((rel.to_string_lossy().into_owned(), text));
            }
        }
    }

    /// Check `src` and hand back what the features operate on.
    fn check(analyzer: &Analyzer, src: &str) -> (Checked, LineIndex) {
        let analysis = analyzer.analyze(src);
        assert!(
            analysis.diagnostics.is_empty(),
            "the fixture does not check: {:?}",
            analysis.diagnostics.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
        (analysis.checked.expect("a clean file produces a check"), LineIndex::new(src))
    }

    /// The position of a substring, for writing fixtures without counting columns.
    fn at(index: &LineIndex, src: &str, needle: &str) -> Position {
        index.position(src.find(needle).unwrap_or_else(|| panic!("`{needle}` is in the fixture")))
    }

    const FIXTURE: &str = r##"use std::io::println;

/// Add two numbers.
/// The second line of the documentation.
fn add(a: i64, b: i64) -> i64 { a + b }

fn main() {
    let total = add(1, 2);
    println("#{total}");
}
"##;

    #[test]
    fn hover_on_a_call_shows_the_signature_and_its_documentation() {
        let a = analyzer();
        let (mut c, idx) = check(&a, FIXTURE);
        let (content, _) = hover(&a, &mut c, &idx, at(&idx, FIXTURE, "add(1"))
            .expect("a call has a hover");
        assert!(content.value.contains("fn add(a: i64, b: i64) -> i64"), "{}", content.value);
        // Both lines, and not the `use` line above them.
        assert!(content.value.contains("Add two numbers."), "{}", content.value);
        assert!(content.value.contains("The second line"), "{}", content.value);
    }

    #[test]
    fn hover_on_a_stdlib_name_shows_where_it_came_from() {
        let a = analyzer();
        let (mut c, idx) = check(&a, FIXTURE);
        let (content, _) = hover(&a, &mut c, &idx, at(&idx, FIXTURE, "println(\""))
            .expect("a stdlib call has a hover");
        assert!(content.value.contains("fn println(s: str)"), "{}", content.value);
        assert!(content.value.contains("std::io"), "{}", content.value);
    }

    /// The case that motivated hovering patterns at all: a binding carries no type of its
    /// own, so this only works because the type is read off the initialiser.
    #[test]
    fn hover_on_a_binding_shows_its_inferred_type() {
        let a = analyzer();
        let (mut c, idx) = check(&a, FIXTURE);
        let (content, _) = hover(&a, &mut c, &idx, at(&idx, FIXTURE, "total ="))
            .expect("a binding has a hover");
        assert!(content.value.contains("i64"), "{}", content.value);
    }

    #[test]
    fn definition_of_a_stdlib_name_points_into_the_stdlib() {
        let a = analyzer();
        let (c, idx) = check(&a, FIXTURE);
        let uri: Uri = "file:///fixture.neon".parse().expect("the fixture URI parses");
        let loc = definition(&a, &c, &idx, &uri, at(&idx, FIXTURE, "println(\""))
            .expect("a stdlib name has a definition");
        assert!(loc.uri.path().as_str().ends_with("std/io.neon"), "landed at {:?}", loc.uri);
    }

    #[test]
    fn definition_of_a_local_name_stays_in_this_file() {
        let a = analyzer();
        let (c, idx) = check(&a, FIXTURE);
        let uri: Uri = "file:///fixture.neon".parse().expect("the fixture URI parses");
        let loc = definition(&a, &c, &idx, &uri, at(&idx, FIXTURE, "add(1"))
            .expect("a local fn has a definition");
        assert_eq!(loc.uri, uri);
        assert_eq!(loc.range.start.line, 4, "the `fn add` line");
    }

    /// Shadowing is the property a name-matching implementation gets wrong, and getting it
    /// wrong means rename silently edits the wrong occurrences.
    #[test]
    fn references_respect_shadowing() {
        let src = "fn main() {\n    let x = 1;\n    { let x = 2; let y = x; }\n    let z = x;\n}\n";
        let a = analyzer();
        let (c, idx) = check(&a, src);

        let inner = references(&a, &c, &idx, at(&idx, src, "x = 2"));
        // The inner binding and the `x` in `let y = x`, and nothing on lines 1 or 3.
        assert_eq!(inner.len(), 2, "inner `x`: {inner:?}");
        assert!(inner.iter().all(|r| r.start.line == 2), "inner `x` escaped its block: {inner:?}");

        let outer = references(&a, &c, &idx, at(&idx, src, "x = 1"));
        assert_eq!(outer.len(), 2, "outer `x`: {outer:?}");
        assert!(outer.iter().any(|r| r.start.line == 3), "outer `x` missed `let z = x`");
    }

    #[test]
    fn rename_is_refused_for_a_name_defined_in_the_stdlib() {
        let a = analyzer();
        let (c, idx) = check(&a, FIXTURE);
        assert!(
            rename(&a, &c, &idx, at(&idx, FIXTURE, "println(\"")).is_none(),
            "renaming into the stdlib would edit the user's toolchain"
        );
    }

    #[test]
    fn rename_covers_the_binding_and_every_use() {
        let a = analyzer();
        let (c, idx) = check(&a, FIXTURE);
        let edits = rename(&a, &c, &idx, at(&idx, FIXTURE, "total =")).expect("a local can be renamed");
        assert_eq!(edits.len(), 2, "the binding and the one use: {edits:?}");
    }

    #[test]
    fn completion_offers_locals_and_stdlib_functions() {
        let a = analyzer();
        let (mut c, idx) = check(&a, FIXTURE);
        let items = completions(&a, &mut c, &idx, at(&idx, FIXTURE, "println(\""));
        let labels: Vec<&str> = items.iter().map(|i| i.label.as_str()).collect();
        assert!(labels.contains(&"total"), "the local in scope is missing");
        assert!(labels.contains(&"add"), "the file's own function is missing");
        // And the detail is the signature, which is the point of offering it.
        let add = items.iter().find(|i| i.label == "add").expect("`add` is offered");
        assert_eq!(add.detail.as_deref(), Some("fn add(a: i64, b: i64) -> i64"));
    }

    #[test]
    fn semantic_tokens_tell_parameters_from_locals_from_functions() {
        let a = analyzer();
        let (c, idx) = check(&a, FIXTURE);
        let toks = semantic_tokens(&c, &idx);
        let kinds: Vec<u32> = toks.iter().map(|t| t.token_type).collect();
        assert!(kinds.contains(&TOK_PARAMETER), "no parameter token: {kinds:?}");
        assert!(kinds.contains(&TOK_FUNCTION), "no function token: {kinds:?}");
        assert!(kinds.contains(&TOK_VARIABLE), "no variable token: {kinds:?}");
    }

    /// The encoding is deltas, and a client renders garbage from the first out-of-order
    /// token onward rather than reporting an error — so this asserts what nothing else
    /// would catch.
    #[test]
    fn semantic_tokens_are_emitted_in_position_order() {
        let a = analyzer();
        let (c, idx) = check(&a, FIXTURE);
        for t in semantic_tokens(&c, &idx) {
            if t.delta_line == 0 {
                // Within a line, a delta of zero would mean two tokens at one place.
                assert!(t.delta_start > 0 || t.length == 0, "non-advancing token: {t:?}");
            }
        }
    }

    #[test]
    fn a_selection_range_widens_from_the_binding_outwards() {
        let a = analyzer();
        let (c, idx) = check(&a, FIXTURE);
        let chain = selection_range(&c, &idx, at(&idx, FIXTURE, "total ="));
        assert!(chain.len() >= 2, "expected to widen at least once: {chain:?}");
        // Strictly widening, which is what makes repeated presses terminate.
        for pair in chain.windows(2) {
            let (inner, outer) = (pair[0], pair[1]);
            assert!(
                outer.start <= inner.start && outer.end >= inner.end,
                "{outer:?} does not contain {inner:?}"
            );
        }
    }

    #[test]
    fn inlay_hints_are_only_for_bindings_that_did_not_write_a_type() {
        let src = "fn main() {\n    let a = 1;\n    let b: i64 = 2;\n}\n";
        let a = analyzer();
        let (mut c, idx) = check(&a, src);
        let hints = inlay_hints(&mut c, &idx);
        assert_eq!(hints.len(), 1, "only the un-annotated `let`: {hints:?}");
        assert_eq!(hints[0].1, ": i64");
    }

    /// A local inside `mod foo` is recorded with module path `["foo"]`, not `[]` — the
    /// checker names the module the code was *written* in. Anything keying "is this the
    /// document I am editing" off an empty module path gets this wrong.
    #[test]
    fn navigation_works_inside_a_nested_module() {
        let src = "mod inner {\n    fn f() -> i64 {\n        let x = 1;\n        x\n    }\n}\n";
        let a = analyzer();
        let (c, idx) = check(&a, src);
        let uri: Uri = "file:///fixture.neon".parse().expect("the fixture URI parses");

        let loc = definition(&a, &c, &idx, &uri, at(&idx, src, "x\n    }"))
            .expect("a local inside a mod has a definition");
        assert_eq!(loc.uri, uri, "it must point back at this file, not the stdlib");

        assert!(
            rename(&a, &c, &idx, at(&idx, src, "x = 1")).is_some(),
            "a local inside a mod is defined in this file and must be renameable"
        );
    }

    #[test]
    fn document_symbols_nest_a_module_inside_itself() {
        let src = "mod inner { fn f() {} }\nfn g() {}\n";
        let a = analyzer();
        let (c, idx) = check(&a, src);
        let syms = document_symbols(&c, &idx);
        let m = syms.iter().find(|s| s.name == "inner").expect("the module is listed");
        assert_eq!(m.children.len(), 1, "the module's function is nested inside it");
        assert_eq!(m.children[0].name, "f");
        assert!(syms.iter().any(|s| s.name == "g"), "the top-level function is listed");
    }

    // ---- doc_above ----
    //
    // A pure function over the trivia table, so these need no stdlib and no check.

    fn docs_for(src: &str, needle: &str) -> Option<String> {
        let lexed = neon_compiler::lexer::lex_full(src).expect("the fixture lexes");
        let at = src.find(needle).expect("the needle is in the fixture");
        doc_above(src, &lexed.trivia, &(at..at + needle.len()))
    }

    #[test]
    fn a_doc_run_is_joined_in_source_order() {
        let d = docs_for("/// one\n/// two\nfn f() {}", "fn f").expect("there is documentation");
        assert_eq!(d, "one\ntwo");
    }

    /// The failure this guards: without it, every function inherits the documentation of
    /// whatever was declared above it, which is worse than no documentation because it is
    /// confidently wrong.
    #[test]
    fn a_blank_line_ends_the_run() {
        assert_eq!(docs_for("/// for the other one\n\nfn f() {}", "fn f"), None);
    }

    #[test]
    fn an_ordinary_comment_is_not_documentation() {
        assert_eq!(docs_for("// just a note\nfn f() {}", "fn f"), None);
    }

    #[test]
    fn a_declaration_with_nothing_above_it_has_no_documentation() {
        assert_eq!(docs_for("fn f() {}", "fn f"), None);
    }
}
