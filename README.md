# neon-lsp

The language server for [Neon](https://github.com/neon-tube/neon). A synchronous,
single-threaded `lsp-server` loop over the compiler's checker — every feature is a query
against a checked module, never a re-derivation. Split out of the main repo; the compiler
comes in as a git dependency (`neon-compiler`), pinned by `Cargo.lock`.

## Build & install

```sh
cargo build --release          # target/release/neon-lsp
# or install straight from git:
cargo install --git https://github.com/neon-tube/neon-lsp
```

The editor plugins ([neon-vscode](https://github.com/neon-tube/neon-vscode),
[neon-zed](https://github.com/neon-tube/neon-zed),
[neon-neovim](https://github.com/neon-tube/neon-neovim)) launch `neon-lsp` from `PATH`, so
make sure the built binary is on it. The server needs `NEON_SYSROOT` set (or a resolvable
`neon` toolchain) for anything past lexer/parser diagnostics.

## Capabilities

Diagnostics, hover, go-to-definition, find-references, document symbols, semantic tokens,
folding ranges, selection ranges, signature help, and completion — advertised in the
`initialize` response, which is where editors read them from.
