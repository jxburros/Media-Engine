//! The CLIP BPE tokenizer, as transformers 4.49's slow `CLIPTokenizer` behaves without `ftfy`:
//! added/special tokens are split out first (leftmost-longest), every other piece goes through
//! BERT's BasicTokenizer (control-character removal, whitespace cleanup, CJK spacing, NFC,
//! lowercasing; no accent stripping, no punctuation splitting), CLIP's regex, the byte-to-unicode
//! map and BPE with `</w>` word ends; then truncation to `model_max_length` − 2, `bos … eos`, and
//! padding to `model_max_length` with the bundle's pad token.

use std::collections::HashMap;
use std::path::Path;

use crate::json::{self, Value};
use crate::unicode;
use crate::Error;

const START: &str = "<|startoftext|>";
const END: &str = "<|endoftext|>";

#[derive(Clone, Debug)]
struct AddedToken {
    content: String,
    id: u32,
    lstrip: bool,
    rstrip: bool,
}

#[derive(Clone, Debug)]
pub struct ClipTokenizer {
    encoder: HashMap<String, u32>,
    ranks: HashMap<(String, String), usize>,
    /// Sorted longest first so the split can take the longest match at a position.
    added: Vec<AddedToken>,
    byte_encoder: [char; 256],
    unk_id: Option<u32>,
    pub bos_id: u32,
    pub eos_id: u32,
    pub pad_id: u32,
    pub max_length: usize,
    pad_left: bool,
    truncate_left: bool,
}

fn read_json(path: &Path) -> Result<Value, Error> {
    let text = std::fs::read_to_string(path).map_err(|e| Error::Failed(format!("tokenizer file: {e}")))?;
    json::parse(&text).map_err(|e| Error::Failed(format!("tokenizer file: {e}")))
}

fn token_text(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Object(_) => value.get("content").and_then(Value::as_str).map(str::to_string),
        _ => None,
    }
}

/// GPT-2's reversible byte → printable-character map.
pub fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut assigned = [false; 256];
    for b in (b'!'..=b'~').chain(0xA1..=0xAC).chain(0xAE..=0xFF) {
        table[b as usize] = char::from_u32(b as u32).unwrap();
        assigned[b as usize] = true;
    }
    let mut n = 0;
    for b in 0..256 {
        if !assigned[b] {
            table[b] = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }
    table
}

impl ClipTokenizer {
    /// Read `vocab.json`, `merges.txt` and the tokenizer config files from a component folder.
    pub fn from_dir(dir: &Path) -> Result<ClipTokenizer, Error> {
        let vocab = read_json(&dir.join("vocab.json"))?;
        let mut encoder = HashMap::new();
        for (token, id) in vocab.as_object().ok_or_else(|| Error::Failed("tokenizer vocabulary".into()))? {
            let id = id.as_i64().filter(|i| *i >= 0).ok_or_else(|| Error::Failed("tokenizer vocabulary".into()))?;
            encoder.insert(token.clone(), id as u32);
        }
        let merges = std::fs::read_to_string(dir.join("merges.txt"))
            .map_err(|e| Error::Failed(format!("tokenizer merges: {e}")))?;
        // `read().strip().split("\n")[1 : 49152 - 256 - 2 + 1]`: the first line is always skipped.
        let mut ranks = HashMap::new();
        for (rank, line) in merges.trim().split('\n').skip(1).take(49152 - 256 - 2).enumerate() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                ranks.insert((parts[0].to_string(), parts[1].to_string()), rank);
            }
        }

        let config_path = dir.join("tokenizer_config.json");
        let config = if config_path.exists() { read_json(&config_path)? } else { Value::Object(Vec::new()) };
        if let Some(class) = config.get("tokenizer_class").and_then(Value::as_str) {
            if class != "CLIPTokenizer" && class != "CLIPTokenizerFast" {
                return Err(Error::Unsupported("tokenizer_class".into()));
            }
        }
        if config.get("split_special_tokens").and_then(Value::as_bool) == Some(true) {
            return Err(Error::Unsupported("split_special_tokens".into()));
        }
        let mut specials: HashMap<&str, String> = HashMap::new();
        let defaults = [("bos_token", START), ("eos_token", END), ("unk_token", END), ("pad_token", END)];
        for (key, default) in defaults {
            specials.insert(key, default.to_string());
        }
        let mut added: Vec<AddedToken> = Vec::new();
        let mut extra = Vec::new();
        let overlay = |source: &Value, specials: &mut HashMap<&str, String>| -> Result<(), Error> {
            for (key, _) in defaults {
                if let Some(value) = source.get(key) {
                    if value.is_null() {
                        continue;
                    }
                    specials.insert(key, token_text(value).ok_or_else(|| Error::Unsupported(key.into()))?);
                }
            }
            if let Some(list) = source.get("additional_special_tokens").and_then(Value::as_array) {
                if !list.is_empty() {
                    return Err(Error::Unsupported("additional_special_tokens".into()));
                }
            }
            Ok(())
        };
        overlay(&config, &mut specials)?;
        if let Some(decoder) = config.get("added_tokens_decoder") {
            for (id, token) in decoder.as_object().ok_or_else(|| Error::Unsupported("added_tokens_decoder".into()))? {
                let id: u32 = id.parse().map_err(|_| Error::Unsupported("added_tokens_decoder".into()))?;
                let content = token_text(token).ok_or_else(|| Error::Unsupported("added_tokens_decoder".into()))?;
                let flag = |name: &str| token.get(name).and_then(Value::as_bool).unwrap_or(false);
                if flag("single_word") {
                    return Err(Error::Unsupported("added_tokens_decoder single_word".into()));
                }
                added.retain(|t| t.content != content);
                added.push(AddedToken { content, id, lstrip: flag("lstrip"), rstrip: flag("rstrip") });
            }
        } else {
            // Legacy layout: special_tokens_map.json and added_tokens.json describe the tokens.
            let map_path = dir.join("special_tokens_map.json");
            if map_path.exists() {
                overlay(&read_json(&map_path)?, &mut specials)?;
            }
            let added_path = dir.join("added_tokens.json");
            if added_path.exists() {
                let legacy = read_json(&added_path)?;
                let mut entries: Vec<(String, u32)> = Vec::new();
                for (content, id) in legacy.as_object().ok_or_else(|| Error::Unsupported("added_tokens.json".into()))? {
                    let id = id.as_i64().ok_or_else(|| Error::Unsupported("added_tokens.json".into()))?;
                    entries.push((content.clone(), id as u32));
                }
                entries.sort_by_key(|e| e.1);
                for (content, id) in entries {
                    extra.push(AddedToken { content, id, lstrip: false, rstrip: false });
                }
            }
        }
        added.extend(extra);
        // Special tokens not yet added are added now: their vocabulary id, else the next id.
        let mut next_id = (encoder.len() + added.iter().filter(|t| !encoder.contains_key(&t.content)).count()) as u32;
        for key in ["bos_token", "eos_token", "unk_token", "pad_token"] {
            let content = specials[key].clone();
            if added.iter().any(|t| t.content == content) {
                continue;
            }
            let id = match encoder.get(&content) {
                Some(&id) => id,
                None => {
                    next_id += 1;
                    next_id - 1
                }
            };
            added.push(AddedToken { content, id, lstrip: false, rstrip: false });
        }
        added.sort_by_key(|t| std::cmp::Reverse(t.content.chars().count()));

        let lookup = |content: &str| -> Option<u32> {
            added.iter().find(|t| t.content == content).map(|t| t.id).or_else(|| encoder.get(content).copied())
        };
        let id_of = |key: &str| lookup(&specials[key]).ok_or_else(|| Error::Unsupported(key.into()));
        let max_length = config
            .get("model_max_length")
            .and_then(Value::as_i64)
            .filter(|n| (2..=4096).contains(n))
            .ok_or_else(|| Error::Unsupported("model_max_length".into()))? as usize;
        let side = |key: &str| -> Result<bool, Error> {
            match config.get(key).and_then(Value::as_str) {
                None | Some("right") => Ok(false),
                Some("left") => Ok(true),
                Some(_) => Err(Error::Unsupported(key.into())),
            }
        };
        Ok(ClipTokenizer {
            unk_id: encoder.get(&specials["unk_token"]).copied(),
            bos_id: id_of("bos_token")?,
            eos_id: id_of("eos_token")?,
            pad_id: id_of("pad_token")?,
            pad_left: side("padding_side")?,
            truncate_left: side("truncation_side")?,
            max_length,
            encoder,
            ranks,
            added,
            byte_encoder: bytes_to_unicode(),
        })
    }

    /// The token strings for `text` (no special tokens added), as `CLIPTokenizer.tokenize`.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        for (piece, added) in self.split_added(text) {
            match added {
                Some(index) => out.push(self.added[index].content.clone()),
                None => {
                    if !piece.is_empty() {
                        self.tokenize_piece(&piece, &mut out);
                    }
                }
            }
        }
        out
    }

    fn token_id(&self, token: &str) -> Result<u32, Error> {
        if let Some(t) = self.added.iter().find(|t| t.content == token) {
            return Ok(t.id);
        }
        self.encoder
            .get(token)
            .copied()
            .or(self.unk_id)
            .ok_or_else(|| Error::Failed("token outside the vocabulary and no unknown token".into()))
    }

    /// `tokenizer(text, padding="max_length", max_length=model_max_length, truncation=True).input_ids`.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        let mut ids = self.tokenize(text).iter().map(|t| self.token_id(t)).collect::<Result<Vec<_>, _>>()?;
        let room = self.max_length - 2;
        if ids.len() > room {
            let remove = ids.len() - room;
            if self.truncate_left {
                ids.drain(..remove);
            } else {
                ids.truncate(room);
            }
        }
        let mut out = Vec::with_capacity(self.max_length);
        out.push(self.bos_id);
        out.extend(ids);
        out.push(self.eos_id);
        let padding = self.max_length - out.len();
        if self.pad_left {
            let mut padded = vec![self.pad_id; padding];
            padded.extend(out);
            out = padded;
        } else {
            out.extend(std::iter::repeat_n(self.pad_id, padding));
        }
        Ok(out)
    }

    /// The added-token trie split, then each added token's lstrip/rstrip on its neighbours.
    fn split_added(&self, text: &str) -> Vec<(String, Option<usize>)> {
        let chars: Vec<char> = text.chars().collect();
        let mut pieces: Vec<(String, Option<usize>)> = Vec::new();
        let mut current = String::new();
        let mut i = 0;
        'scan: while i < chars.len() {
            for (index, token) in self.added.iter().enumerate() {
                let n = token.content.chars().count();
                if n > 0 && i + n <= chars.len() && token.content.chars().eq(chars[i..i + n].iter().copied()) {
                    pieces.push((std::mem::take(&mut current), None));
                    pieces.push((token.content.clone(), Some(index)));
                    i += n;
                    continue 'scan;
                }
            }
            current.push(chars[i]);
            i += 1;
        }
        pieces.push((current, None));
        for k in 0..pieces.len() {
            if let Some(index) = pieces[k].1 {
                let token = &self.added[index];
                if token.rstrip && k + 1 < pieces.len() && pieces[k + 1].1.is_none() {
                    pieces[k + 1].0 = pieces[k + 1].0.trim_start_matches(unicode::is_py_space).to_string();
                }
                if token.lstrip && k > 0 && pieces[k - 1].1.is_none() {
                    pieces[k - 1].0 = pieces[k - 1].0.trim_end_matches(unicode::is_py_space).to_string();
                }
            }
        }
        pieces
    }

    fn tokenize_piece(&self, text: &str, out: &mut Vec<String>) {
        let words = basic_tokenize(text);
        let text: Vec<char> = words.join(" ").chars().collect();
        for word in clip_regex(&text) {
            let mut encoded = String::new();
            for b in word.bytes() {
                encoded.push(self.byte_encoder[b as usize]);
            }
            out.extend(self.bpe(&encoded));
        }
    }

    fn bpe(&self, token: &str) -> Vec<String> {
        if token == START || token == END {
            return vec![token.to_string()];
        }
        let chars: Vec<char> = token.chars().collect();
        let mut word: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
        let last = word.pop().unwrap();
        word.push(last + "</w>");
        if word.len() < 2 {
            return word;
        }
        loop {
            let mut best: Option<(usize, usize)> = None;
            for k in 0..word.len() - 1 {
                if let Some(&rank) = self.ranks.get(&(word[k].clone(), word[k + 1].clone())) {
                    if best.is_none_or(|(r, _)| rank < r) {
                        best = Some((rank, k));
                    }
                }
            }
            let Some((_, k)) = best else { break };
            let (first, second) = (word[k].clone(), word[k + 1].clone());
            let mut merged = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == first && word[i + 1] == second {
                    merged.push(format!("{first}{second}"));
                    i += 2;
                } else {
                    merged.push(word[i].clone());
                    i += 1;
                }
            }
            word = merged;
            if word.len() == 1 {
                break;
            }
        }
        word
    }
}

/// Python `str.split()`.
fn py_split(text: &str) -> Vec<String> {
    text.split(unicode::is_py_space).filter(|s| !s.is_empty()).map(str::to_string).collect()
}

/// `BasicTokenizer(strip_accents=False, do_split_on_punc=False).tokenize`.
fn basic_tokenize(text: &str) -> Vec<String> {
    let mut cleaned = String::with_capacity(text.len());
    for c in text.chars() {
        if c == '\0' || c == '\u{FFFD}' || unicode::is_control(c) {
            continue;
        }
        if unicode::is_whitespace(c) {
            cleaned.push(' ');
        } else if unicode::is_chinese(c) {
            cleaned.push(' ');
            cleaned.push(c);
            cleaned.push(' ');
        } else {
            cleaned.push(c);
        }
    }
    let normalized = unicode::nfc(&cleaned);
    let lowered: Vec<String> = py_split(&normalized).iter().map(|t| unicode::py_lower(t)).collect();
    py_split(&lowered.join(" "))
}

/// Simple case folding for the pattern's literal letters (`regex.IGNORECASE`).
fn fold(c: char) -> char {
    match c {
        '\u{17F}' => 's',
        'A'..='Z' => c.to_ascii_lowercase(),
        _ => c,
    }
}

fn literal_at(text: &[char], i: usize, literal: &str) -> bool {
    let n = literal.chars().count();
    i + n <= text.len() && literal.chars().zip(&text[i..i + n]).all(|(l, &c)| fold(c) == l)
}

/// `re.findall(r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+", text, re.IGNORECASE)`.
fn clip_regex(text: &[char]) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < text.len() {
        let mut end = None;
        for literal in [START, END, "'s", "'t", "'re", "'ve", "'m", "'ll", "'d"] {
            if literal_at(text, i, literal) {
                end = Some(i + literal.chars().count());
                break;
            }
        }
        if end.is_none() && unicode::is_re_letter(text[i]) {
            let mut j = i;
            while j < text.len() && unicode::is_re_letter(text[j]) {
                j += 1;
            }
            end = Some(j);
        }
        if end.is_none() && unicode::is_re_number(text[i]) {
            end = Some(i + 1);
        }
        if end.is_none() && unicode::is_re_other(text[i]) {
            let mut j = i;
            while j < text.len() && unicode::is_re_other(text[j]) {
                j += 1;
            }
            end = Some(j);
        }
        match end {
            Some(j) => {
                out.push(text[i..j].iter().collect());
                i = j;
            }
            None => i += 1,
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Token ids from transformers 4.49's CLIPTokenizer (no ftfy) for the fixture built by
    /// tests/make_tokenizer_fixture.py: punctuation, contractions, CJK, emoji, NFC, control and
    /// whitespace cleanup, Final_Sigma, added-token splitting (including tokenizer_2's `!` pad),
    /// and prompts longer than 77 tokens.
    #[test]
    fn token_ids_match_transformers() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/clip_tokenizer");
        let expected = json::parse(&std::fs::read_to_string(root.join("expected.json")).unwrap()).unwrap();
        let tokenizers: Vec<(&str, ClipTokenizer)> = ["tokenizer", "tokenizer_2"]
            .iter()
            .map(|name| (*name, ClipTokenizer::from_dir(&root.join(name)).unwrap()))
            .collect();
        assert_eq!(tokenizers[1].1.pad_id, 0);
        let mut checked = 0;
        for row in expected.as_array().unwrap() {
            let text = row.get("text").unwrap().as_str().unwrap();
            for (name, tokenizer) in &tokenizers {
                let want: Vec<u32> =
                    row.get(name).unwrap().as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as u32).collect();
                let got = tokenizer.encode(text).unwrap();
                assert_eq!(got, want, "{name}: {text:?} tokens {:?}", tokenizer.tokenize(text));
                checked += 1;
            }
        }
        assert_eq!(checked, 24);
    }

    #[test]
    fn byte_map_is_reversible() {
        let table = bytes_to_unicode();
        let mut seen: Vec<char> = table.to_vec();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 256);
        assert_eq!(table[b' ' as usize], '\u{120}');
    }
}
