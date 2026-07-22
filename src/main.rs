//! The Neon language server.
//!
//! Every capability advertised here works. An editor told a server can do hover, and then
//! given nothing back, looks broken in a way the user cannot diagnose — so a capability
//! appears in `ServerCapabilities` only once it answers.
//!
//! **Where the answers come from.** Not from an index this server builds and maintains.
//! The type checker already decides, for every expression, what its type is and which
//! binding each name refers to; it records both in a `TypecheckResult`. That value used to
//! be dropped on the floor the moment the diagnostics were extracted from it — which is
//! why an earlier version of this comment described go-to-definition as blocked on "a
//! span-to-definition index" that did not exist. It did exist. Keeping the result instead
//! of discarding it (`analysis::Checked`) turned eight capabilities from "not built" into
//! "read the map", and made none of them cost a second pass over the source.
//!
//! So every feature in `features.rs` is a query, never a derivation: position to byte
//! offset, offset to the innermost AST node, node id to whatever the checker recorded
//! against it. A hover that recomputed a type would be a second type checker, and two type
//! checkers disagree.
//!
//! The loop is synchronous and handles one message at a time. That is affordable because
//! the two expensive things have been taken off the per-keystroke path: the stdlib is
//! parsed once per session (`analysis::Analyzer`), and checks are debounced so a burst of
//! typing costs one check rather than one per character. Requests that read the analysis
//! force a pending check first — see `needs_analysis` — because otherwise the answer to
//! the first hover after an edit is "nothing here".

// Clippy objects to `Uri` as a `HashMap` key because `fluent_uri`, which `lsp_types`
// wraps, holds a `Cell` internally. It is a parse cache and nothing else: `lsp_types`
// defines both `Hash` and `PartialEq` for `Uri` on `as_str()` alone (`lsp-types-0.97.0`,
// `src/uri.rs:68` and `:76`), so neither can observe the cell, and a key's hash cannot
// drift while it sits in the map. Keying by `String` instead would sidestep the lint at
// the cost of converting on every lookup and rebuilding a `Uri` for every `Location`.
#![allow(clippy::mutable_key_type)]

mod analysis;
mod features;
mod position;
mod toolchain;

use lsp_server::{Connection, ExtractError, Message, Notification, Request, RequestId, Response};
use lsp_types::notification::{
    DidChangeTextDocument, DidCloseTextDocument, DidOpenTextDocument, DidSaveTextDocument,
    LogMessage, Notification as _, PublishDiagnostics, ShowMessage,
};
use lsp_types::request::{
    Completion, DocumentSymbolRequest, FoldingRangeRequest, Formatting, GotoDefinition,
    HoverRequest, InlayHintRequest, References, Rename, Request as _, SelectionRangeRequest,
    SemanticTokensFullRequest, SignatureHelpRequest,
};
use lsp_types::{
    CompletionOptions, CompletionResponse, DiagnosticRelatedInformation, DiagnosticSeverity,
    DocumentFormattingParams, DocumentSymbol, DocumentSymbolResponse, GotoDefinitionResponse, Hover,
    HoverContents, HoverProviderCapability, InitializeParams, InlayHint, InlayHintKind,
    InlayHintLabel, Location, LogMessageParams, MessageType, OneOf, PublishDiagnosticsParams,
    FoldingRange, FoldingRangeKind, Range, SelectionRange, SemanticTokens, SemanticTokensFullOptions,
    SemanticTokensLegend, SemanticTokensOptions, SemanticTokensResult,
    SemanticTokensServerCapabilities, ServerCapabilities, ShowMessageParams, SignatureHelpOptions,
    TextDocumentSyncCapability, TextDocumentSyncKind, TextEdit, Uri, WorkspaceEdit,
};
use position::LineIndex;
use std::collections::HashMap;
use std::error::Error;
use std::time::{Duration, Instant};

/// How long the server waits for typing to stop before checking.
///
/// A check is a blocking pass over the user's module, and a keystroke invalidates the one
/// in flight anyway, so running per character spends the whole budget on results nobody
/// reads. 150ms is below the threshold where a pause feels like lag and comfortably above
/// a fast typist's inter-key interval, so ordinary typing produces exactly one check when
/// the user stops.
const DEBOUNCE: Duration = Duration::from_millis(150);

/// The longest a document may go unchecked while still being edited.
///
/// Without this, continuous typing renews the debounce forever and diagnostics never
/// appear at all — the failure mode where the feature looks broken rather than slow.
const MAX_DELAY: Duration = Duration::from_millis(750);

/// One open document: its current text, and the most recent check of it.
///
/// `checked` deliberately survives a failed parse. A file is unparseable for most of the
/// time anyone is typing in it, and a server that dropped its analysis on every
/// half-written line would answer "no hover, no definition" precisely when the user is
/// reaching for them. So the last check that succeeded stays until a later one replaces
/// it — slightly stale beats absent, and the diagnostics (which are recomputed every
/// time) are what tell the user the file is currently broken.
struct Doc {
    index: LineIndex,
    checked: Option<analysis::Checked>,
}

/// Open documents, by URI. The editor's copy is authoritative — a file on disk may be
/// stale by several keystrokes, and checking the stale one would report errors the user
/// has already fixed.
type Documents = HashMap<Uri, Doc>;

/// A document that has changed and not yet been checked.
struct Pending {
    uri: Uri,
    /// When it first went dirty — the anchor for `MAX_DELAY`.
    first: Instant,
    /// When it last changed — the anchor for `DEBOUNCE`.
    last: Instant,
}

impl Pending {
    /// The moment the check should run: whichever of the two limits comes first.
    fn deadline(&self) -> Instant {
        (self.last + DEBOUNCE).min(self.first + MAX_DELAY)
    }
}

fn main() -> Result<(), Box<dyn Error + Sync + Send>> {
    // stdio: the transport every editor supports without configuration.
    let (connection, io_threads) = Connection::stdio();

    let capabilities = serde_json::to_value(ServerCapabilities {
        // Full sync: the checker needs a whole module anyway, so applying incremental
        // edits would only be bookkeeping on the way to reassembling the same string.
        text_document_sync: Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::FULL)),
        document_formatting_provider: Some(OneOf::Left(true)),
        hover_provider: Some(HoverProviderCapability::Simple(true)),
        definition_provider: Some(OneOf::Left(true)),
        references_provider: Some(OneOf::Left(true)),
        rename_provider: Some(OneOf::Left(true)),
        document_symbol_provider: Some(OneOf::Left(true)),
        inlay_hint_provider: Some(OneOf::Left(true)),
        completion_provider: Some(CompletionOptions {
            // `::` because a qualified name is the case where completion has to fire
            // without a fresh identifier character to trigger on. `.` is deliberately
            // absent: Neon has no method-call syntax, so a dot is field access on a record
            // whose fields are already offered by the identifier path.
            trigger_characters: Some(vec![":".into()]),
            ..Default::default()
        }),
        signature_help_provider: Some(SignatureHelpOptions {
            trigger_characters: Some(vec!["(".into(), ",".into()]),
            ..Default::default()
        }),
        folding_range_provider: Some(lsp_types::FoldingRangeProviderCapability::Simple(true)),
        selection_range_provider: Some(lsp_types::SelectionRangeProviderCapability::Simple(true)),
        semantic_tokens_provider: Some(
            SemanticTokensServerCapabilities::SemanticTokensOptions(SemanticTokensOptions {
                legend: SemanticTokensLegend {
                    token_types: features::TOKEN_TYPES.to_vec(),
                    token_modifiers: Vec::new(),
                },
                // Whole-file only. Range and delta requests exist to avoid re-tokenising a
                // large file on every edit, but the tokens here are a by-product of a check
                // that already ran, so producing them costs one walk of the AST — less than
                // the bookkeeping a delta would need.
                full: Some(SemanticTokensFullOptions::Bool(true)),
                ..Default::default()
            }),
        ),
        ..Default::default()
    })?;

    let init = connection.initialize(capabilities)?;
    let _params: InitializeParams = serde_json::from_value(init)?;

    let analyzer = start_analyzer(&connection)?;
    serve(&connection, &analyzer)?;

    // Dropping the connection first is required, not tidiness: `join` waits for the
    // writer thread, and that thread only finishes once its end of the channel is
    // disconnected. Holding `connection` across the join deadlocks the server after a
    // clean shutdown — the editor gets its response and the process never exits, leaking
    // one per session.
    drop(connection);
    io_threads.join()?;
    Ok(())
}

/// Load the toolchain's stdlib, telling the user whatever happened.
///
/// A server with no stdlib can still lex and parse, which is worth having, so this
/// degrades rather than refusing to start — but it does not degrade *quietly*. The old
/// behaviour was to fall back in silence, which left the user looking at a file with no
/// diagnostics that would not compile, and nothing anywhere to explain why. That is worse
/// than either extreme: a clean file is an assertion, and the server was making one it
/// could not support.
///
/// So both channels are used, for different readers. `window/logMessage` gets the full
/// detail and lands in the editor's output pane, which is where someone debugging a setup
/// looks; it is sent on success too, so "which stdlib am I actually checking against" has
/// an answer without reproducing a failure. `window/showMessage` fires only on trouble and
/// only once, because the user has to be interrupted at least once — nobody opens the log
/// pane to discover a problem they do not yet know they have — and a popup per keystroke
/// would train them to dismiss it unread.
fn start_analyzer(
    connection: &Connection,
) -> Result<analysis::Analyzer, Box<dyn Error + Sync + Send>> {
    let (analyzer, log, warn) = match toolchain::load() {
        Ok(std) => {
            let detail = format!(
                "neon-lsp: loaded {} stdlib file(s) from '{}'; type checking is on.",
                std.sources.len(),
                std.dir.display()
            );
            match analysis::Analyzer::new(&std.dir, &std.sources) {
                Ok(a) => (a, detail, None),
                // A stdlib that does not parse is a broken toolchain, not the user's
                // fault — but they still need to know why type errors stopped appearing.
                Err(e) => {
                    let msg = format!(
                        "neon-lsp: the stdlib at '{}' did not parse, so only syntax errors \
                         will appear. {e}",
                        std.dir.display()
                    );
                    (analysis::Analyzer::syntax_only(), msg.clone(), Some(msg))
                }
            }
        }
        Err(failure) => {
            let msg = failure.message();
            (analysis::Analyzer::syntax_only(), msg.clone(), Some(msg))
        }
    };

    notify(
        connection,
        LogMessage::METHOD,
        LogMessageParams {
            typ: if warn.is_some() { MessageType::WARNING } else { MessageType::INFO },
            message: log,
        },
    )?;
    if let Some(message) = warn {
        notify(
            connection,
            ShowMessage::METHOD,
            ShowMessageParams { typ: MessageType::WARNING, message },
        )?;
    }
    Ok(analyzer)
}

fn notify<P: serde::Serialize>(
    connection: &Connection,
    method: &str,
    params: P,
) -> Result<(), Box<dyn Error + Sync + Send>> {
    let note = Notification::new(method.to_string(), params);
    connection.sender.send(Message::Notification(note))?;
    Ok(())
}

fn serve(
    connection: &Connection,
    analyzer: &analysis::Analyzer,
) -> Result<(), Box<dyn Error + Sync + Send>> {
    let mut docs: Documents = HashMap::new();
    let mut pending: Option<Pending> = None;

    loop {
        // With a check pending, wait only until its deadline; without one, wait forever.
        // This is the whole debounce: every message that arrives first renews the wait,
        // and the timeout firing is what makes the check happen. No threads and no
        // cancellation are needed, because the work never starts until the wait ends.
        let msg = match &pending {
            Some(p) => {
                let wait = p.deadline().saturating_duration_since(Instant::now());
                match connection.receiver.recv_timeout(wait) {
                    Ok(msg) => msg,
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                        flush(connection, &mut docs, analyzer, pending.take())?;
                        continue;
                    }
                    // The editor vanished. Its diagnostics do not matter now.
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => return Ok(()),
                }
            }
            None => match connection.receiver.recv() {
                Ok(msg) => msg,
                Err(_) => return Ok(()),
            },
        };

        match msg {
            Message::Request(req) => {
                if connection.handle_shutdown(&req)? {
                    return Ok(());
                }
                // A request that reads the analysis must not race the debounce. Without
                // this, opening a file and hovering inside the next 150ms answers
                // "nothing here" — the check has not run yet — and the user sees a
                // server that works only if they pause first. Formatting is exempt
                // because it reprints from the text, which is always current, and
                // completion fires per keystroke in some editors, so forcing a check for
                // every request would undo the debounce entirely.
                if needs_analysis(&req.method) {
                    let for_this_doc = pending.as_ref().is_some_and(|p| Some(&p.uri) == uri_of(&req).as_ref());
                    if for_this_doc {
                        flush(connection, &mut docs, analyzer, pending.take())?;
                    }
                }
                let response = handle_request(req, &mut docs, analyzer);
                connection.sender.send(Message::Response(response))?;
            }
            Message::Notification(note) => match document_event(note) {
                Some(Event::Changed(uri, text)) => {
                    let index = LineIndex::new(&text);
                    match docs.get_mut(&uri) {
                        // Keep the previous check: the new text may not parse, and this
                        // is exactly the moment its answers are still worth having.
                        Some(doc) => doc.index = index,
                        None => {
                            docs.insert(uri.clone(), Doc { index, checked: None });
                        }
                    }
                    let now = Instant::now();
                    match &mut pending {
                        // Same document still settling: renew the debounce, but keep the
                        // original `first` so `MAX_DELAY` still bounds the total wait.
                        Some(p) if p.uri == uri => p.last = now,
                        // A different document went dirty. Check the old one now rather
                        // than letting edits elsewhere postpone it indefinitely.
                        Some(_) => {
                            flush(connection, &mut docs, analyzer, pending.take())?;
                            pending = Some(Pending { uri, first: now, last: now });
                        }
                        None => pending = Some(Pending { uri, first: now, last: now }),
                    }
                }
                Some(Event::Closed(uri)) => {
                    docs.remove(&uri);
                    if pending.as_ref().is_some_and(|p| p.uri == uri) {
                        pending = None;
                    }
                    // An editor keeps showing the last set it was told about, so a closed
                    // file would otherwise leave stale errors in the problems panel
                    // forever. Not debounced: closing is a deliberate act, not a burst.
                    connection.sender.send(Message::Notification(empty_diagnostics(&uri)))?;
                }
                None => {}
            },
            Message::Response(_) => {}
        }
    }
}

/// Run the deferred check and publish its result.
fn flush(
    connection: &Connection,
    docs: &mut Documents,
    analyzer: &analysis::Analyzer,
    pending: Option<Pending>,
) -> Result<(), Box<dyn Error + Sync + Send>> {
    let Some(p) = pending else { return Ok(()) };
    // Closed between the edit and the deadline: nothing to say about it.
    let Some(doc) = docs.get_mut(&p.uri) else { return Ok(()) };
    connection.sender.send(Message::Notification(publish(&p.uri, doc, analyzer)))?;
    Ok(())
}

/// What a text-document notification means to this server.
enum Event {
    Changed(Uri, String),
    Closed(Uri),
}

fn document_event(note: Notification) -> Option<Event> {
    match note.method.as_str() {
        DidOpenTextDocument::METHOD => {
            let p: lsp_types::DidOpenTextDocumentParams = serde_json::from_value(note.params).ok()?;
            Some(Event::Changed(p.text_document.uri, p.text_document.text))
        }
        DidChangeTextDocument::METHOD => {
            let p: lsp_types::DidChangeTextDocumentParams =
                serde_json::from_value(note.params).ok()?;
            // Full sync, so the last change carries the whole document.
            let text = p.content_changes.into_iter().next_back()?.text;
            Some(Event::Changed(p.text_document.uri, text))
        }
        DidSaveTextDocument::METHOD => {
            let p: lsp_types::DidSaveTextDocumentParams = serde_json::from_value(note.params).ok()?;
            Some(Event::Changed(p.text_document.uri, p.text?))
        }
        DidCloseTextDocument::METHOD => {
            let p: lsp_types::DidCloseTextDocumentParams =
                serde_json::from_value(note.params).ok()?;
            Some(Event::Closed(p.text_document.uri))
        }
        _ => None,
    }
}

fn empty_diagnostics(uri: &Uri) -> Notification {
    Notification::new(
        PublishDiagnostics::METHOD.to_string(),
        PublishDiagnosticsParams { uri: uri.clone(), diagnostics: Vec::new(), version: None },
    )
}

/// Check a document, keep the result, and build the notification carrying its diagnostics.
///
/// The check is the expensive part of the server and it now produces two things rather
/// than one: the diagnostics, which go out immediately, and the `Checked`, which stays
/// behind to answer hover and navigation. Doing both from one pass is the point — the
/// alternative is re-checking the file the first time someone hovers over it.
fn publish(uri: &Uri, doc: &mut Doc, analyzer: &analysis::Analyzer) -> Notification {
    let index = &doc.index;
    let analysis = analyzer.analyze(index.text());
    let diagnostics = analysis
        .diagnostics
        .into_iter()
        .map(|d| lsp_types::Diagnostic {
            range: Range {
                start: index.position(d.span.start),
                end: index.position(d.span.end),
            },
            severity: Some(DiagnosticSeverity::ERROR),
            source: Some("neon".into()),
            message: match d.help {
                // The help text is the actionable half of most of these messages, and an
                // editor shows one string. Appending beats dropping it.
                Some(help) => format!("{}\n\nhelp: {help}", d.message),
                None => d.message,
            },
            related_information: (!d.labels.is_empty()).then(|| {
                d.labels
                    .into_iter()
                    .map(|(span, note)| DiagnosticRelatedInformation {
                        location: Location {
                            uri: uri.clone(),
                            range: Range {
                                start: index.position(span.start),
                                end: index.position(span.end),
                            },
                        },
                        message: note,
                    })
                    .collect()
            }),
            ..Default::default()
        })
        .collect::<Vec<_>>();

    // Only on success. A failed parse leaves the previous check in place; see `Doc`.
    if analysis.checked.is_some() {
        doc.checked = analysis.checked;
    }

    Notification::new(
        PublishDiagnostics::METHOD.to_string(),
        PublishDiagnosticsParams { uri: uri.clone(), diagnostics, version: None },
    )
}

/// Whether answering this request means reading a `Checked`.
///
/// Formatting is the one handled request that does not: it reprints from the document
/// text, which every `didChange` updates synchronously.
fn needs_analysis(method: &str) -> bool {
    matches!(
        method,
        HoverRequest::METHOD
            | GotoDefinition::METHOD
            | References::METHOD
            | Rename::METHOD
            | Completion::METHOD
            | SignatureHelpRequest::METHOD
            | DocumentSymbolRequest::METHOD
            | InlayHintRequest::METHOD
            | SemanticTokensFullRequest::METHOD
            | FoldingRangeRequest::METHOD
            | SelectionRangeRequest::METHOD
    )
}

/// The document a request is about, read straight off the wire.
///
/// Every request handled here carries its URI in one of two shapes, and this only has to
/// decide whether a pending check is for the *same* document — so a miss is a missed
/// optimisation, not a wrong answer.
fn uri_of(req: &Request) -> Option<Uri> {
    req.params
        .get("textDocument")
        .and_then(|d| d.get("uri"))
        .and_then(|u| u.as_str())
        .and_then(|u| u.parse().ok())
}

/// Dispatch one request.
///
/// Every arm but formatting needs a `Checked`, and takes `&mut` to get one: printing a
/// type interns its complement, so rendering an answer mutates the type table. Nothing
/// observable changes — the table is hash-consed — but the borrow checker is right that
/// it is a mutation, and threading `&mut` is cheaper than an interior-mutability wrapper
/// that would hide it.
fn handle_request(req: Request, docs: &mut Documents, analyzer: &analysis::Analyzer) -> Response {
    match req.method.as_str() {
        Formatting::METHOD => match cast::<Formatting>(req) {
            Ok((id, params)) => format_document(id, params, docs),
            Err(err) => err,
        },

        HoverRequest::METHOD => answer::<HoverRequest>(req, docs, |doc, uri, pos| {
            let checked = doc.checked.as_mut()?;
            let (contents, range) = features::hover(analyzer, checked, &doc.index, pos)?;
            let _ = uri;
            Some(Hover { contents: HoverContents::Markup(contents), range: Some(range) })
        }),

        GotoDefinition::METHOD => answer::<GotoDefinition>(req, docs, |doc, uri, pos| {
            let checked = doc.checked.as_ref()?;
            let loc = features::definition(analyzer, checked, &doc.index, uri, pos)?;
            Some(GotoDefinitionResponse::Scalar(loc))
        }),

        References::METHOD => answer::<References>(req, docs, |doc, uri, pos| {
            let checked = doc.checked.as_ref()?;
            let ranges = features::references(analyzer, checked, &doc.index, pos);
            Some(
                ranges
                    .into_iter()
                    .map(|range| Location { uri: uri.clone(), range })
                    .collect::<Vec<_>>(),
            )
        }),

        Rename::METHOD => answer_rename(req, docs, analyzer),

        Completion::METHOD => answer::<Completion>(req, docs, |doc, uri, pos| {
            let _ = uri;
            let checked = doc.checked.as_mut()?;
            Some(CompletionResponse::Array(features::completions(
                analyzer, checked, &doc.index, pos,
            )))
        }),

        SignatureHelpRequest::METHOD => answer::<SignatureHelpRequest>(req, docs, |doc, uri, pos| {
            let _ = uri;
            features::signature_help(doc.checked.as_mut()?, &doc.index, pos)
        }),

        DocumentSymbolRequest::METHOD => match cast::<DocumentSymbolRequest>(req) {
            Ok((id, params)) => {
                let symbols = docs
                    .get(&params.text_document.uri)
                    .and_then(|doc| doc.checked.as_ref().map(|c| (c, &doc.index)))
                    .map(|(c, index)| {
                        features::document_symbols(c, index).iter().map(to_symbol).collect()
                    })
                    .unwrap_or_default();
                Response::new_ok(id, DocumentSymbolResponse::Nested(symbols))
            }
            Err(err) => err,
        },

        InlayHintRequest::METHOD => match cast::<InlayHintRequest>(req) {
            Ok((id, params)) => {
                let hints = docs
                    .get_mut(&params.text_document.uri)
                    .and_then(|doc| {
                        let index = &doc.index;
                        doc.checked.as_mut().map(|c| features::inlay_hints(c, index))
                    })
                    .unwrap_or_default()
                    .into_iter()
                    .map(|(position, label)| InlayHint {
                        position,
                        label: InlayHintLabel::String(label),
                        kind: Some(InlayHintKind::TYPE),
                        text_edits: None,
                        tooltip: None,
                        padding_left: None,
                        padding_right: None,
                        data: None,
                    })
                    .collect::<Vec<_>>();
                Response::new_ok(id, hints)
            }
            Err(err) => err,
        },

        SemanticTokensFullRequest::METHOD => match cast::<SemanticTokensFullRequest>(req) {
            Ok((id, params)) => {
                let data = docs
                    .get(&params.text_document.uri)
                    .and_then(|doc| doc.checked.as_ref().map(|c| (c, &doc.index)))
                    .map(|(c, index)| features::semantic_tokens(c, index))
                    .unwrap_or_default();
                Response::new_ok(
                    id,
                    SemanticTokensResult::Tokens(SemanticTokens { result_id: None, data }),
                )
            }
            Err(err) => err,
        },

        FoldingRangeRequest::METHOD => match cast::<FoldingRangeRequest>(req) {
            Ok((id, params)) => {
                let ranges = docs
                    .get(&params.text_document.uri)
                    .and_then(|doc| doc.checked.as_ref().map(|c| (c, &doc.index)))
                    .map(|(c, index)| features::folding_ranges(c, index))
                    .unwrap_or_default()
                    .into_iter()
                    .map(|(start_line, end_line)| FoldingRange {
                        start_line,
                        end_line,
                        kind: Some(FoldingRangeKind::Region),
                        ..Default::default()
                    })
                    .collect::<Vec<_>>();
                Response::new_ok(id, ranges)
            }
            Err(err) => err,
        },

        SelectionRangeRequest::METHOD => match cast::<SelectionRangeRequest>(req) {
            Ok((id, params)) => {
                let Some((checked, index)) = docs
                    .get(&params.text_document.uri)
                    .and_then(|doc| doc.checked.as_ref().map(|c| (c, &doc.index)))
                else {
                    return Response::new_ok(id, serde_json::Value::Null);
                };
                // One answer per position asked about, in the same order.
                let out: Vec<SelectionRange> = params
                    .positions
                    .iter()
                    .filter_map(|&pos| chain(features::selection_range(checked, index, pos)))
                    .collect();
                Response::new_ok(id, out)
            }
            Err(err) => err,
        },

        _ => Response::new_err(
            req.id,
            lsp_server::ErrorCode::MethodNotFound as i32,
            format!("unhandled request: {}", req.method),
        ),
    }
}

/// Nested containing ranges, innermost first, as the linked list the protocol wants.
///
/// Built back to front because each link owns its parent: the outermost range is the only
/// one with no parent, so it has to exist before the one inside it can point at it.
fn chain(mut ranges: Vec<Range>) -> Option<SelectionRange> {
    let outermost = ranges.pop()?;
    let mut node = SelectionRange { range: outermost, parent: None };
    while let Some(range) = ranges.pop() {
        node = SelectionRange { range, parent: Some(Box::new(node)) };
    }
    Some(node)
}

/// The shape every position-taking request shares: find the document, run the query, and
/// answer `null` when there is nothing to say.
///
/// `null` rather than an error, because "no hover here" is the ordinary case — the cursor
/// spends most of its life on whitespace — and an editor that got an error response for it
/// would log a failure every time the mouse moved.
fn answer<R>(
    req: Request,
    docs: &mut Documents,
    f: impl FnOnce(&mut Doc, &Uri, lsp_types::Position) -> R::Result,
) -> Response
where
    R: lsp_types::request::Request,
    R::Params: serde::de::DeserializeOwned + HasPosition,
    R::Result: serde::Serialize + Default,
{
    let (id, params) = match cast::<R>(req) {
        Ok(v) => v,
        Err(err) => return err,
    };
    let (uri, pos) = params.at();
    // Every one of these requests has an `Option` result, whose `Default` is `None` and
    // serialises to `null` — which is exactly the "nothing here" answer, so an unknown
    // document needs no special case.
    let result = docs.get_mut(&uri).map(|doc| f(doc, &uri, pos)).unwrap_or_default();
    Response::new_ok(id, result)
}

/// Rename is its own arm because it is the one request that answers "no" meaningfully.
///
/// Declining a rename has to be an *error*, not an empty edit: an empty `WorkspaceEdit`
/// tells the editor the rename succeeded and changed nothing, so the user sees their
/// symbol keep its old name with no explanation. The error message is what tells them the
/// definition lives in the stdlib.
fn answer_rename(req: Request, docs: &mut Documents, analyzer: &analysis::Analyzer) -> Response {
    let (id, params) = match cast::<Rename>(req) {
        Ok(v) => v,
        Err(err) => return err,
    };
    let uri = params.text_document_position.text_document.uri.clone();
    let pos = params.text_document_position.position;

    let ranges = docs
        .get(&uri)
        .and_then(|doc| doc.checked.as_ref().map(|c| (c, &doc.index)))
        .and_then(|(c, index)| features::rename(analyzer, c, index, pos));

    let Some(ranges) = ranges else {
        return Response::new_err(
            id,
            lsp_server::ErrorCode::InvalidRequest as i32,
            "this name cannot be renamed from here: it is not defined in this file".into(),
        );
    };

    let edits: Vec<TextEdit> =
        ranges.into_iter().map(|range| TextEdit { range, new_text: params.new_name.clone() }).collect();
    let changes = std::collections::HashMap::from([(uri, edits)]);
    Response::new_ok(id, WorkspaceEdit { changes: Some(changes), ..Default::default() })
}

/// One outline entry, converted for the wire.
///
/// `selection_range` is the same as `range` here: it is meant to be the identifier alone,
/// which would need a span the AST does not record separately for every declaration kind.
/// Pointing at the whole declaration is a worse highlight than pointing at its name, but
/// it is never a *wrong* one, and the protocol requires the selection range to be
/// contained by the range.
#[allow(deprecated)] // `DocumentSymbol::deprecated` is required by the struct literal.
fn to_symbol(s: &features::Symbol) -> DocumentSymbol {
    DocumentSymbol {
        name: s.name.clone(),
        detail: None,
        kind: s.kind,
        tags: None,
        deprecated: None,
        range: s.range,
        selection_range: s.range,
        children: Some(s.children.iter().map(to_symbol).collect()),
    }
}

/// The document and position a request names.
///
/// LSP spells this three different ways across the requests handled here, so the trait
/// exists to let `answer` be written once rather than once per request type.
trait HasPosition {
    fn at(&self) -> (Uri, lsp_types::Position);
}

impl HasPosition for lsp_types::TextDocumentPositionParams {
    fn at(&self) -> (Uri, lsp_types::Position) {
        (self.text_document.uri.clone(), self.position)
    }
}

impl HasPosition for lsp_types::HoverParams {
    fn at(&self) -> (Uri, lsp_types::Position) {
        self.text_document_position_params.at()
    }
}

impl HasPosition for lsp_types::GotoDefinitionParams {
    fn at(&self) -> (Uri, lsp_types::Position) {
        self.text_document_position_params.at()
    }
}

impl HasPosition for lsp_types::SignatureHelpParams {
    fn at(&self) -> (Uri, lsp_types::Position) {
        self.text_document_position_params.at()
    }
}

impl HasPosition for lsp_types::ReferenceParams {
    fn at(&self) -> (Uri, lsp_types::Position) {
        self.text_document_position.at()
    }
}

impl HasPosition for lsp_types::CompletionParams {
    fn at(&self) -> (Uri, lsp_types::Position) {
        self.text_document_position.at()
    }
}

/// Format a whole document, as one edit replacing everything.
///
/// A file that does not parse is left alone: the formatter reprints from the AST, so
/// there is nothing to reprint, and returning no edits is exactly right — the editor
/// leaves the buffer as the user typed it. Failing loudly here would mean an error popup
/// on every format-on-save while a line is half-written.
fn format_document(id: RequestId, params: DocumentFormattingParams, docs: &Documents) -> Response {
    let Some(index) = docs.get(&params.text_document.uri).map(|d| &d.index) else {
        return Response::new_ok(id, Vec::<TextEdit>::new());
    };
    let Ok(formatted) = neon_compiler::format::format(index.text()) else {
        return Response::new_ok(id, Vec::<TextEdit>::new());
    };
    if formatted == index.text() {
        return Response::new_ok(id, Vec::<TextEdit>::new());
    }
    let end = index.position(index.text().len());
    let edit = TextEdit {
        range: Range { start: lsp_types::Position { line: 0, character: 0 }, end },
        new_text: formatted,
    };
    Response::new_ok(id, vec![edit])
}

fn cast<R>(req: Request) -> Result<(RequestId, R::Params), Response>
where
    R: lsp_types::request::Request,
    R::Params: serde::de::DeserializeOwned,
{
    req.extract(R::METHOD).map_err(|e| match e {
        ExtractError::MethodMismatch(r) => Response::new_err(
            r.id,
            lsp_server::ErrorCode::MethodNotFound as i32,
            "method mismatch".into(),
        ),
        ExtractError::JsonError { method, error } => Response::new_err(
            RequestId::from(0),
            lsp_server::ErrorCode::InvalidParams as i32,
            format!("bad params for {method}: {error}"),
        ),
    })
}
