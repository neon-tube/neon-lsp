//! Byte offsets to LSP positions.
//!
//! The compiler counts bytes; LSP counts UTF-16 code units within a line, which is neither
//! bytes nor characters. Getting this wrong is invisible in ASCII and silently misplaces
//! every span after the first non-ASCII character in a file — an underline that drifts a
//! little further right with each emoji. Neon's strings are unvalidated bytes and its
//! corpus already contains multi-byte text, so this is not a hypothetical.
//!
//! The index is built once per document version and binary-searched, rather than scanning
//! from the start for each diagnostic: a file with many errors is exactly the file being
//! edited, and quadratic behaviour there would be felt.

use lsp_types::Position;

/// Line start offsets for one document, for turning byte offsets into positions.
pub struct LineIndex {
    /// Byte offset of the first character of each line. Always starts with 0, so a
    /// document with no newlines still has one entry.
    starts: Vec<usize>,
    text: String,
}

impl LineIndex {
    pub fn new(text: &str) -> LineIndex {
        let mut starts = vec![0];
        starts.extend(text.match_indices('\n').map(|(i, _)| i + 1));
        LineIndex { starts, text: text.to_string() }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    /// The position of a byte offset. An offset past the end clamps to the end of the
    /// document: a span that runs off the end is a compiler bug, but dropping the
    /// diagnostic or panicking in an editor session is a worse answer than pointing at
    /// the last character.
    pub fn position(&self, offset: usize) -> Position {
        let offset = offset.min(self.text.len());
        let line = self.starts.partition_point(|&s| s <= offset) - 1;
        let line_start = self.starts[line];
        // Only the part of the line before the offset counts, and it counts in UTF-16.
        let prefix = &self.text[line_start..offset];
        let character = prefix.chars().map(char::len_utf16).sum::<usize>();
        Position { line: line as u32, character: character as u32 }
    }

    /// The byte offset of a position — `position` run backwards.
    ///
    /// Every request that names a place in the document rather than reporting one arrives
    /// this way round: hover, go-to-definition, completion and rename all say "line 12,
    /// character 7" and mean a byte offset the compiler can compare against a span. The
    /// UTF-16 walk is the same one `position` does, stopped when the budget runs out
    /// instead of when the offset is reached.
    ///
    /// Out-of-range input clamps rather than failing. An editor can legitimately ask about
    /// a position one past the end of a line (that is where the cursor sits after the last
    /// character), and a client racing a document change can ask about a line that no
    /// longer exists. Neither deserves an error response.
    pub fn offset(&self, pos: Position) -> usize {
        let Some(&line_start) = self.starts.get(pos.line as usize) else {
            return self.text.len();
        };
        // The line's own text, so a character index past its end stops at the newline
        // rather than running on into the next line.
        let line_end = self
            .starts
            .get(pos.line as usize + 1)
            .map_or(self.text.len(), |&next| next);

        let mut units = 0usize;
        for (i, c) in self.text[line_start..line_end].char_indices() {
            if units >= pos.character as usize {
                return line_start + i;
            }
            units += c.len_utf16();
        }
        line_end
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_multibyte_line_counts_utf16_units_not_bytes() {
        // "é" is two bytes and one UTF-16 unit; the emoji is four bytes and *two* units,
        // being outside the basic plane. A byte count would report 6 here and a character
        // count 2, and both would be wrong.
        let idx = LineIndex::new("é🙂x");
        let offset = "é🙂".len();
        assert_eq!(idx.position(offset), Position { line: 0, character: 3 });
    }

    #[test]
    fn positions_are_relative_to_their_own_line() {
        let idx = LineIndex::new("ab\ncd\nef");
        assert_eq!(idx.position(4), Position { line: 1, character: 1 });
        assert_eq!(idx.position(6), Position { line: 2, character: 0 });
    }

    #[test]
    fn an_offset_past_the_end_clamps() {
        let idx = LineIndex::new("ab\n");
        assert_eq!(idx.position(999), idx.position("ab\n".len()));
    }

    /// The property that matters for every position-taking request: asking for the
    /// position of an offset and then the offset of that position gets back where it
    /// started. A drift of one here puts hover on the character next to the cursor.
    #[test]
    fn offsets_and_positions_round_trip() {
        let text = "fn f() {\n  let é = \"🙂\";\n  é\n}\n";
        let idx = LineIndex::new(text);
        for offset in text.char_indices().map(|(i, _)| i) {
            assert_eq!(idx.offset(idx.position(offset)), offset, "offset {offset} drifted");
        }
    }

    #[test]
    fn a_character_index_counts_utf16_units_not_bytes() {
        // Same fixture as the forward test: the emoji is two UTF-16 units, so character
        // 3 is the `x` at byte 6. A byte-counting implementation would land mid-emoji.
        let idx = LineIndex::new("é🙂x");
        assert_eq!(idx.offset(Position { line: 0, character: 3 }), "é🙂".len());
    }

    #[test]
    fn a_character_past_the_end_of_a_line_stops_at_the_line_end() {
        // Where the cursor sits after the last character of a line. It must not run on
        // into the next line, or hover at end-of-line would report the wrong statement.
        let idx = LineIndex::new("ab\ncd");
        assert_eq!(idx.offset(Position { line: 0, character: 99 }), 3);
    }

    #[test]
    fn a_line_that_does_not_exist_clamps_to_the_end() {
        let idx = LineIndex::new("ab\n");
        assert_eq!(idx.offset(Position { line: 99, character: 0 }), 3);
    }
}
