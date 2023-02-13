use std::{
    collections::HashSet,
    error::Error,
    fs::read_to_string,
    iter::{repeat_with, Skip, Take},
    str::Chars,
    time::Instant,
};

use ukr::NotUkrainian;

type DictImpl<K, V> = std::collections::BTreeMap<K, V>;
// type DictImpl<K, V> = std::collections::HashMap<K, V>;
type VecImpl<V> = smallvec::SmallVec<[V; 3]>;
// type VecImpl<V> = Vec<V>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dictionary<'a>(Vec<DictImpl<u128, VecImpl<&'a str>>>);

impl<'a> Dictionary<'a> {
    fn new(fuziness: usize) -> Self {
        Dictionary(repeat_with(DictImpl::new).take(fuziness).collect())
    }
    fn buckets(&self) -> std::slice::Iter<'_, DictImpl<u128, VecImpl<&'a str>>> {
        self.0.iter()
    }
    fn buckets_mut(&mut self) -> std::slice::IterMut<'_, DictImpl<u128, VecImpl<&'a str>>> {
        self.0.iter_mut()
    }
    fn fuziness(&self) -> usize {
        self.0.len()
    }
}

fn insert_masked<'a>(dict: &mut Dictionary<'a>, word: &'a str) -> Result<(), NotUkrainian> {
    let fuziness = dict.fuziness();
    for (i, bucket) in dict.buckets_mut().enumerate() {
        let mut repeats = HashSet::with_capacity(word.len() + (fuziness - 1) * 2 + 1);

        for masked in masked_words(word, i + 1) {
            if let Ok(hash) = ukr::hash(masked) {
                if !repeats.insert(hash) {
                    continue;
                }
                bucket.entry(hash).or_default().push(word);
            }
        }
    }

    Ok(())
}

fn build_dictionary(input: &str, fuziness: usize) -> Result<Dictionary<'_>, NotUkrainian> {
    let mut dict = Dictionary::new(fuziness);
    let line_count = input.chars().filter(|ch| *ch == '\n').count();
    let mut last_measure = std::time::Instant::now();

    for (idx, word) in input.split('\n').enumerate() {
        insert_masked(&mut dict, word)?;

        if idx % 100_000 == 0 {
            let now = std::time::Instant::now();
            let took = (now - last_measure).as_millis();
            last_measure = now;
            println!("Processed {idx} out of {line_count}, took {took} ms")
        }
    }
    Ok(dict)
}

fn all_masked_words(
    fuziness: usize,
    word: &str,
) -> impl Iterator<Item = impl Iterator<Item = MaskedWordIter<'_>>> {
    (1..=fuziness).map(|i| masked_words(word, i))
}

fn masked_words(word: &str, fuziness: usize) -> impl Iterator<Item = MaskedWordIter<'_>> {
    let char_count = word.chars().count();

    let start_gap = (1..fuziness).map(|i| MaskedWordIter::start_strip(word, i));
    let end_gap = (1..fuziness)
        .rev()
        .map(|i| MaskedWordIter::end_strip(word, i));
    let max_padding = char_count.checked_sub(fuziness).unwrap_or(0);
    let middle_gaps = (0..=max_padding).map(move |i| MaskedWordIter::combined(word, i, fuziness));

    std::iter::once(MaskedWordIter::whole(word))
        .chain(start_gap)
        .chain(middle_gaps)
        .chain(end_gap)
}

pub fn fuzzy_lookup<'a, 'b>(
    dict: &'b Dictionary<'a>,
    word: &'b str,
) -> impl Iterator<Item = Result<HashSet<&'a str>, NotUkrainian>> + 'b {
    dict.buckets()
        .zip(all_masked_words(dict.fuziness(), word))
        .enumerate()
        .map(|(fuzziness, (bucket, masked))| {
            let mut result = HashSet::new();

            for word in masked {
                let word: String = word.collect();
                let hash = ukr::hash(word.chars())?;
                if let Some(words) = bucket.get(&hash) {
                    println!(
                        "match! fuzziness: {fuzziness}, {word} with hash {hash} -> {:?}",
                        words
                    );
                    result.extend(words);
                }
            }

            Ok(result)
        })
}

struct GapIter<T> {
    iter: T,
    offset: usize,
    len: usize,
}

impl<T: Iterator> Iterator for GapIter<T> {
    type Item = T::Item;

    fn next(&mut self) -> Option<Self::Item> {
        match (self.offset, self.len) {
            (0, 0) => self.iter.next(),
            (0, _) => loop {
                if self.len == 0 {
                    return self.iter.next();
                }
                if self.iter.next().is_none() {
                    return None;
                }
                self.len -= 1;
            },
            (_, _) => {
                self.offset -= 1;
                self.iter.next()
            }
        }
    }
}

enum MaskedWordIter<'a> {
    Whole(Chars<'a>),
    StartStrip(Skip<Chars<'a>>),
    EndStrip(Take<Chars<'a>>),
    Combined(GapIter<Chars<'a>>),
}

impl<'a> Iterator for MaskedWordIter<'a> {
    type Item = char;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            MaskedWordIter::Whole(iter) => iter.next(),
            MaskedWordIter::StartStrip(iter) => iter.next(),
            MaskedWordIter::EndStrip(iter) => iter.next(),
            MaskedWordIter::Combined(iter) => iter.next(),
        }
    }
}

impl<'a> MaskedWordIter<'a> {
    fn whole(word: &'a str) -> Self {
        Self::Whole(word.chars())
    }
    fn start_strip(word: &'a str, amount: usize) -> Self {
        Self::StartStrip(word.chars().skip(amount))
    }
    fn end_strip(word: &'a str, amount: usize) -> Self {
        let char_count = word.chars().count().checked_sub(amount).unwrap_or(0);
        Self::EndStrip(word.chars().take(char_count))
    }
    fn combined(word: &'a str, start: usize, len: usize) -> Self {
        Self::Combined(GapIter {
            iter: word.chars(),
            offset: start,
            len,
        })
    }
}

#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

fn main() -> Result<(), Box<dyn Error>> {
    // let words: Vec<_> = "абрикос авокадо камбоджа кембрідж кокос"
    //     .split_whitespace()
    //     .collect();
    // let hashes: Vec<_> = words
    //     .iter()
    //     .cloned()
    //     .map(str::chars)
    //     .map(ukr::hash)
    //     .map(Result::unwrap)
    //     // .map(|hash| format!("{:#016b}", hash >> 64))
    //     .collect();
    // // assert!(is_sorted(&hashes.unwrap()));
    // println!("{:#?}", hashes);

    // return Ok(());

    // let mut dict = Dictionary::new(3);
    // insert_masked(&mut dict, "слон")?;
    // insert_masked(&mut dict, "масло")?;
    // println!("{:#?}", dict);

    // let res: Result<Vec<_>, _> = fuzzy_lookup(&dict, "мати")
    //     .enumerate()
    //     .map(|(i, res)| res.map(|v| (i, v)))
    //     .collect();

    // println!("{:?}", res?);

    // let a: Vec<_> = all_masked_words(3, "мати")
    //     .enumerate()
    //     .flat_map(|(i, words)| {
    //         words.map(move |word| {
    //             let word: String = word.collect();
    //             let hash = ukr::hash(word.chars()).unwrap();
    //             (i, word, hash)
    //         })
    //     })
    //     .collect();

    // println!("{:#?}", a);

    // let res: Result<Vec<_>, _> = fuzzy_lookup(&dict, "масл")
    //     .enumerate()
    //     .map(|(i, res)| res.map(|v| (i, v)))
    //     .collect();

    // println!("{:?}", res?);

    let input_str = read_to_string("../dict_uk/out/words_spell.txt")?;
    let start = Instant::now();
    let _dict = build_dictionary(&input_str, 1)?;
    let took = Instant::now() - start;
    println!("took: {}", took.as_millis());
    // mem_print();

    // let mut counts = BTreeMap::<usize, u32>::new();
    // for bucket in dict.buckets() {
    //     for subbucket_len in bucket.values().map(VecImpl::len) {
    //         *counts.entry(subbucket_len).or_default() += 1;
    //     }
    // }
    // println!("{:#?}", counts);

    // let res: Result<Vec<_>, _> = fuzzy_lookup(&dict, "маши")
    //     .enumerate()
    //     .map(|(i, res)| res.map(|v| (i, v)))
    //     .collect();

    // println!("{:#?}", res);

    Ok(())
}

mod ukr {
    use std::{error::Error, fmt::Display};

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct NotUkrainian;

    impl Display for NotUkrainian {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(
                "Provided text contains characters not from Ukrainian alphabet, or is too long",
            )
        }
    }

    impl Error for NotUkrainian {}

    const MAX_WORD_COUNT: usize = 20;

    const ALPHABET: [char; 64] = [
        'а', 'б', 'в', 'г', 'ґ', 'д', 'е', 'є', 'ж', 'з', 'и', 'і', 'ї', 'й', 'к', 'л', 'м', 'н',
        'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'x', 'ц', 'ч', 'ш', 'щ', 'ь', 'ю', 'я', 'А', 'Б', 'В',
        'Г', 'Д', 'Е', 'Є', 'Ж', 'З', 'И', 'І', 'Ї', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т',
        'У', 'Ф', 'X', 'Ч', 'Ш', 'Щ', 'Ю', 'Я', '-', '\'',
    ];

    pub fn hash(word: impl IntoIterator<Item = char>) -> Result<u128, NotUkrainian> {
        let mut count = 0;
        let mut hash: u128 = 0;
        for char in word {
            if count > MAX_WORD_COUNT {
                return Err(NotUkrainian);
            }
            if char == 'ґ' {
                continue;
            }
            let position = ALPHABET
                .iter()
                .position(|&c| c == char)
                .ok_or(NotUkrainian)?;
            hash |= (position as u128) << (122 - count * 6);
            count += 1;
        }
        Ok(hash)
    }
}

#[cfg(test)]
mod test {
    use crate::{masked_words, ukr, GapIter};

    #[test]
    fn hashing_preserves_ordering() {
        let words: Vec<_> = "абрикос авокадо камбоджа кембрідж кокос"
            .split_whitespace()
            .collect();
        let hashes: Vec<_> = words
            .iter()
            .cloned()
            .map(str::chars)
            .map(ukr::hash)
            .map(Result::unwrap)
            .collect();
        assert!(is_sorted(&hashes));
    }

    fn is_sorted<T: PartialOrd>(slice: &[T]) -> bool {
        slice
            .iter()
            .zip(slice[1..].iter())
            .all(|(prev, next)| prev < next)
    }

    #[test]
    fn masked_words_on_one() {
        let res: Vec<String> = masked_words("hello", 1).map(Iterator::collect).collect();
        assert_eq!(res, vec!["hello", "ello", "hllo", "helo", "helo", "hell"]);
    }

    #[test]
    fn gap_iter_middle_gap() {
        let iter = GapIter {
            iter: (1..=10).into_iter(),
            offset: 3,
            len: 5,
        };
        let res: Vec<_> = iter.collect();
        assert_eq!(res, vec![1, 2, 3, 9, 10]);
    }

    #[test]
    fn gap_iter_start_gap() {
        let iter = GapIter {
            iter: (1..=10).into_iter(),
            offset: 0,
            len: 5,
        };
        let res: Vec<_> = iter.collect();
        assert_eq!(res, vec![6, 7, 8, 9, 10]);
    }

    #[test]
    fn gap_past_end_gap() {
        let iter = GapIter {
            iter: (1..=10).into_iter(),
            offset: 8,
            len: 5,
        };
        let res: Vec<_> = iter.collect();
        assert_eq!(res, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn masking_butter() {
        let masked: Vec<String> = masked_words("масло", 3).map(Iterator::collect).collect();
        assert_eq!(masked, vec!["асло", "сло", "ло", "мо", "ма", "мас", "масл"]);
    }
}
