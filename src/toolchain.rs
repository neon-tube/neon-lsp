//! Asking the compiler where its stdlib is, and reading it.
//!
//! The server does not discover the sysroot itself. It runs `neon sysroot --stdlib` once
//! at startup and believes the answer. That is deliberate and it is the whole design of
//! this module: discovery logic linked into the LSP would report the sysroot of *this*
//! binary's idea of the toolchain, which is not necessarily the toolchain the user builds
//! with. Shelling out to whichever `neon` is on PATH means the editor's diagnostics and
//! `neon build` agree by construction, even when the two binaries come from different
//! builds — and there is exactly one copy of the probe order (`NEON_SYSROOT`, dev tree,
//! installed layout) to keep correct.
//!
//! The cost is a process spawn per session and a hard dependency on the CLI having that
//! subcommand. It runs once, at startup, never per keystroke.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Why the server has no stdlib. Carried rather than logged at the point of failure so the
/// protocol layer can decide how loudly to say it — this module has no connection to talk on.
///
/// The two cases are kept apart because they are different user actions: a missing binary
/// is a PATH or installation problem, and a binary that ran and refused is a broken or
/// incomplete sysroot. Collapsing them into "no stdlib" is what made the old server
/// unactionable.
pub enum Failure {
    /// No `neon` executable anywhere this looked.
    NotFound { looked: Vec<PathBuf> },
    /// `neon` ran but did not yield a usable stdlib path.
    Unusable { program: PathBuf, reason: String },
}

impl Failure {
    /// One line naming the cause and what to do about it. Shown to the user verbatim.
    pub fn message(&self) -> String {
        match self {
            Failure::NotFound { looked } => format!(
                "neon-lsp cannot find the `neon` compiler, so it can only report syntax \
                 errors — type errors will not appear. Looked in: {}. \
                 Put `neon` on PATH, or set NEON_LSP_COMPILER to its full path.",
                looked.iter().map(|p| format!("'{}'", p.display())).collect::<Vec<_>>().join(", ")
            ),
            Failure::Unusable { program, reason } => format!(
                "neon-lsp found `{}` but it reported no usable stdlib, so only syntax errors \
                 will appear. {reason} \
                 Set NEON_SYSROOT to a sysroot containing `stdlib/`.",
                program.display()
            ),
        }
    }
}

/// The stdlib source as `(relative path, text)` pairs, plus where it came from.
pub struct Stdlib {
    pub dir: PathBuf,
    pub sources: Vec<(String, String)>,
}

/// Locate `neon`, ask it for the stdlib directory, and read every `.neon` under it.
pub fn load() -> Result<Stdlib, Failure> {
    let (program, looked) = find_compiler();
    let program = program.ok_or(Failure::NotFound { looked })?;

    let output = Command::new(&program)
        .args(["sysroot", "--stdlib"])
        // The reply is parsed, and its errors are shown in an editor popup that cannot
        // render terminal escapes. Asking for no colour is cheaper than stripping it.
        .env("NO_COLOR", "1")
        .output()
        .map_err(|e| Failure::Unusable {
            program: program.clone(),
            reason: format!("Running it failed: {e}."),
        })?;

    if !output.status.success() {
        // The CLI's errors are the actionable ones (which path it probed, what was
        // missing); forwarding its stderr beats inventing a worse message here.
        let detail = condense(&String::from_utf8_lossy(&output.stderr));
        return Err(Failure::Unusable {
            program,
            reason: if detail.is_empty() {
                format!("It exited with {}.", output.status)
            } else {
                format!("It said: {detail}")
            },
        });
    }

    let dir = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim().to_string());
    // Resolved, because this path becomes a `file://` URI in every go-to-definition that
    // lands in the stdlib. The CLI reports it relative to its own executable, so it
    // arrives looking like `.../target/debug/../stdlib` — which is the same directory but
    // not the same string, and an editor asked to open a URI containing `..` may decline,
    // or may open a second buffer onto a file it already has. Failure to canonicalise is
    // not fatal: the unresolved path still reads.
    let dir = dir.canonicalize().unwrap_or(dir);
    if dir.as_os_str().is_empty() {
        return Err(Failure::Unusable {
            program,
            reason: "It printed nothing where a stdlib path was expected.".into(),
        });
    }

    let mut sources = Vec::new();
    collect(&dir, &dir, &mut sources).map_err(|e| Failure::Unusable {
        program: program.clone(),
        reason: format!("Reading the stdlib at '{}' failed: {e}.", dir.display()),
    })?;
    if sources.is_empty() {
        return Err(Failure::Unusable {
            program,
            reason: format!("There are no `.neon` files under '{}'.", dir.display()),
        });
    }
    // Sorted for a stable module order, matching what the CLI's loader does — the same
    // sources in a different order would number `ExprId`s differently for no reason.
    sources.sort();
    Ok(Stdlib { dir, sources })
}

/// The `neon` to ask, and the places that were tried if there was none.
///
/// Order: an explicit override, then next to this executable, then PATH. Side-by-side
/// beats PATH because a toolchain ships `neon` and `neon-lsp` in one directory, and an
/// editor launched from a desktop session often has a PATH that does not include it.
fn find_compiler() -> (Option<PathBuf>, Vec<PathBuf>) {
    let mut looked = Vec::new();

    // The override is taken as given, without an existence check: if someone points this
    // at a path that is wrong, the spawn error names it, which is more useful than this
    // silently falling through to a different compiler than the one they asked for.
    if let Some(p) = std::env::var_os("NEON_LSP_COMPILER") {
        return (Some(PathBuf::from(p)), looked);
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let sibling = dir.join(exe_name());
            looked.push(sibling.clone());
            if sibling.is_file() {
                return (Some(sibling), looked);
            }
        }
    }

    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            let candidate = dir.join(exe_name());
            if candidate.is_file() {
                return (Some(candidate), looked);
            }
        }
        looked.push(PathBuf::from("PATH"));
    }

    (None, looked)
}

/// A multi-line `color_eyre` report squeezed into one line fit for a popup.
///
/// The CLI formats for a terminal: a numbered cause chain, a `Location:` block, and advice
/// about `RUST_BACKTRACE`. Only the causes say anything the user can act on, and an editor
/// notification shows one string, so everything from `Location:` down is dropped and the
/// rest is joined. Escape sequences are stripped as well as suppressed, because `NO_COLOR`
/// is a request the child is free to ignore.
fn condense(stderr: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    for line in strip_ansi(stderr).lines() {
        let line = line.trim();
        // The trailer, and everything after it, is about debugging the compiler.
        if line.starts_with("Location:") || line.starts_with("Backtrace omitted") {
            break;
        }
        // `Error:` alone is a header; `0:` and `1:` number the cause chain.
        let line = line.strip_prefix("Error:").unwrap_or(line).trim();
        // `0: the message` — drop the index, keep the message.
        let line = match line.split_once(": ") {
            Some((n, rest)) if n.chars().all(|c| c.is_ascii_digit()) && !n.is_empty() => rest.trim(),
            _ => line,
        };
        if line.is_empty() {
            continue;
        }
        out.push(line.to_string());
    }
    let joined = out.join(" ");
    // Long enough for the real message, short enough not to fill the screen.
    if joined.chars().count() > 400 {
        joined.chars().take(400).collect::<String>() + "…"
    } else {
        joined
    }
}

/// Remove CSI escape sequences. Narrow by design: this only ever sees the CLI's own
/// output, which uses simple colour codes.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            // `ESC [ ... <final byte in @..~>`; anything else is dropped with the ESC.
            if chars.next() == Some('[') {
                for c in chars.by_ref() {
                    if ('@'..='~').contains(&c) {
                        break;
                    }
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn exe_name() -> &'static str {
    if cfg!(windows) {
        "neon.exe"
    } else {
        "neon"
    }
}

fn collect(root: &Path, dir: &Path, out: &mut Vec<(String, String)>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect(root, &path, out)?;
        } else if path.extension().is_some_and(|e| e == "neon") {
            let rel = path
                .strip_prefix(root)
                .expect("collected under root")
                .to_string_lossy()
                .replace('\\', "/");
            out.push((rel, std::fs::read_to_string(&path)?));
        }
    }
    Ok(())
}
