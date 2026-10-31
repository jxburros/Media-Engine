//! The Unicode predicates the CLIP tokenizer depends on, from tables recorded off the reference
//! Python environment (`tools/gen_unicode_tables.py`): Python's `unicodedata` categories and NFC,
//! `str.lower` / `str.isspace`, and the `regex` module's classes as CLIP's pattern uses them.

use crate::unicode_tables as t;

fn in_ranges(ranges: &[(u32, u32)], c: char) -> bool {
    let cp = c as u32;
    ranges
        .binary_search_by(|&(a, b)| {
            if b < cp {
                std::cmp::Ordering::Less
            } else if a > cp {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// transformers' `_is_control` (category C*, except tab/newline/carriage return).
pub fn is_control(c: char) -> bool {
    in_ranges(t::CONTROL, c)
}

/// transformers' `_is_whitespace` (space, tab, newline, carriage return, or category Zs).
pub fn is_whitespace(c: char) -> bool {
    in_ranges(t::WHITESPACE, c)
}

/// Python `str.isspace` (what `str.split()` / `str.strip()` split and strip on).
pub fn is_py_space(c: char) -> bool {
    in_ranges(t::PY_SPACE, c)
}

/// `[\p{L}]` under `regex.IGNORECASE`.
pub fn is_re_letter(c: char) -> bool {
    in_ranges(t::RE_LETTER, c)
}

/// `[\p{N}]` under `regex.IGNORECASE`.
pub fn is_re_number(c: char) -> bool {
    in_ranges(t::RE_NUMBER, c)
}

/// `[^\s\p{L}\p{N}]` under `regex.IGNORECASE`.
pub fn is_re_other(c: char) -> bool {
    in_ranges(t::RE_OTHER, c)
}

/// The CJK ranges of BERT's `_is_chinese_char`.
pub fn is_chinese(c: char) -> bool {
    let cp = c as u32;
    (0x4E00..=0x9FFF).contains(&cp)
        || (0x3400..=0x4DBF).contains(&cp)
        || (0x20000..=0x2A6DF).contains(&cp)
        || (0x2A700..=0x2B73F).contains(&cp)
        || (0x2B740..=0x2B81F).contains(&cp)
        || (0x2B820..=0x2CEAF).contains(&cp)
        || (0xF900..=0xFAFF).contains(&cp)
        || (0x2F800..=0x2FA1F).contains(&cp)
}

/// Python's `str.lower`, including the Final_Sigma rule for U+03A3.
pub fn py_lower(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    for (i, &c) in chars.iter().enumerate() {
        if c == '\u{3A3}' {
            out.push(if final_sigma(&chars, i) { '\u{3C2}' } else { '\u{3C3}' });
            continue;
        }
        let cp = c as u32;
        if let Ok(k) = t::LOWER.binary_search_by_key(&cp, |&(a, _)| a) {
            out.push(char::from_u32(t::LOWER[k].1).unwrap());
        } else if let Some((_, many)) = t::LOWER_MULTI.iter().find(|(a, _)| *a == cp) {
            out.extend(many.iter().map(|&m| char::from_u32(m).unwrap()));
        } else {
            out.push(c);
        }
    }
    out
}

fn final_sigma(chars: &[char], i: usize) -> bool {
    let ignorable = |c: char| in_ranges(t::CASE_IGNORABLE, c);
    let cased = |c: char| in_ranges(t::CASED, c);
    let mut j = i;
    let mut before = None;
    while j > 0 {
        j -= 1;
        if !ignorable(chars[j]) {
            before = Some(chars[j]);
            break;
        }
    }
    if !before.is_some_and(cased) {
        return false;
    }
    for &c in &chars[i + 1..] {
        if !ignorable(c) {
            return !cased(c);
        }
    }
    true
}

fn ccc(c: char) -> u8 {
    let cp = c as u32;
    match t::CCC.binary_search_by(|&(a, b, _)| {
        if b < cp {
            std::cmp::Ordering::Less
        } else if a > cp {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }) {
        Ok(k) => t::CCC[k].2,
        Err(_) => 0,
    }
}

const S_BASE: u32 = 0xAC00;
const L_BASE: u32 = 0x1100;
const V_BASE: u32 = 0x1161;
const T_BASE: u32 = 0x11A7;
const L_COUNT: u32 = 19;
const V_COUNT: u32 = 21;
const T_COUNT: u32 = 28;
const N_COUNT: u32 = V_COUNT * T_COUNT;
const S_COUNT: u32 = L_COUNT * N_COUNT;

fn decompose_into(c: char, out: &mut Vec<char>) {
    let cp = c as u32;
    if (S_BASE..S_BASE + S_COUNT).contains(&cp) {
        let s = cp - S_BASE;
        out.push(char::from_u32(L_BASE + s / N_COUNT).unwrap());
        out.push(char::from_u32(V_BASE + (s % N_COUNT) / T_COUNT).unwrap());
        if !s.is_multiple_of(T_COUNT) {
            out.push(char::from_u32(T_BASE + s % T_COUNT).unwrap());
        }
        return;
    }
    match t::DECOMP.binary_search_by_key(&cp, |&(a, _, _)| a) {
        Ok(k) => {
            let (_, offset, len) = t::DECOMP[k];
            let offset = offset as usize;
            out.extend(t::DECOMP_DATA[offset..offset + len as usize].iter().map(|&x| char::from_u32(x).unwrap()));
        }
        Err(_) => out.push(c),
    }
}

fn compose_pair(a: char, b: char) -> Option<char> {
    let (a, b) = (a as u32, b as u32);
    if (L_BASE..L_BASE + L_COUNT).contains(&a) && (V_BASE..V_BASE + V_COUNT).contains(&b) {
        return char::from_u32(S_BASE + ((a - L_BASE) * V_COUNT + (b - V_BASE)) * T_COUNT);
    }
    if (S_BASE..S_BASE + S_COUNT).contains(&a)
        && (a - S_BASE).is_multiple_of(T_COUNT)
        && (T_BASE + 1..T_BASE + T_COUNT).contains(&b)
    {
        return char::from_u32(a + (b - T_BASE));
    }
    t::COMPOSE.binary_search_by(|&(x, y, _)| (x, y).cmp(&(a, b))).ok().map(|k| char::from_u32(t::COMPOSE[k].2).unwrap())
}

/// `unicodedata.normalize("NFC", s)`.
pub fn nfc(s: &str) -> String {
    if s.is_ascii() {
        return s.to_string();
    }
    let mut chars = Vec::with_capacity(s.len());
    for c in s.chars() {
        decompose_into(c, &mut chars);
    }
    // Canonical ordering: stable sort of each run of non-starters by combining class.
    let mut i = 0;
    while i < chars.len() {
        if ccc(chars[i]) == 0 {
            i += 1;
            continue;
        }
        let start = i;
        while i < chars.len() && ccc(chars[i]) != 0 {
            i += 1;
        }
        chars[start..i].sort_by_key(|&c| ccc(c));
    }
    // Canonical composition.
    let mut out: Vec<char> = Vec::with_capacity(chars.len());
    let mut starter: Option<usize> = None;
    for c in chars {
        let class = ccc(c);
        if let Some(si) = starter {
            let last = *out.last().unwrap();
            let blocked = out.len() - 1 != si && {
                let last_class = ccc(last);
                last_class == 0 || last_class >= class
            };
            if !blocked {
                if let Some(composite) = compose_pair(out[si], c) {
                    out[si] = composite;
                    continue;
                }
            }
        }
        if class == 0 {
            starter = Some(out.len());
        }
        out.push(c);
    }
    out.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_and_case() {
        assert_eq!(nfc("e\u{301}"), "\u{e9}");
        assert_eq!(nfc("\u{212B}"), "\u{C5}");
        assert_eq!(nfc("\u{1100}\u{1161}\u{11A8}"), "\u{AC01}");
        assert_eq!(nfc("a\u{323}\u{302}"), "\u{1EAD}");
        assert_eq!(py_lower("ΟΔΟΣ ΣΑ"), "οδος σα");
        assert_eq!(py_lower("ΟΔΟΣ"), "οδος");
        assert_eq!(py_lower("\u{130}X"), "i\u{307}x");
        assert!(is_control('\u{200B}') && !is_control('\n') && is_whitespace('\u{3000}'));
        assert!(is_re_letter('é') && is_re_number('٣') && is_re_other('!') && !is_re_other(' '));
    }
}
