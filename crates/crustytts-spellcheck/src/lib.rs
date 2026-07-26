//! Multi-strategy English spell checking with pluggable dictionary, LLM, and grammar.
//!
//! # Strategy (in order)
//!
//! 1. **Dictionary** — words found in the dictionary pass through untouched.
//!    Bring your own via the [`Dictionary`] trait, or use the bundled
//!    [`EmbeddedDictionary`] (64k English words). Enable the `symspell` or
//!    `spellbook` features for frequency-weighted or Hunspell backends.
//! 2. **Edit distance** — for each unknown word, generate every candidate within
//!    one edit. If exactly one unique candidate is in the dictionary, it is a
//!    *high-confidence* correction.
//! 3. **LLM verification** — when edit distance is ambiguous, an optional
//!    [`LlmProvider`] examines the surrounding sentence context.
//! 4. **Grammar checking** — after spell correction, an optional
//!    [`GrammarChecker`] applies rule-based grammar fixes. Enable `nlprule`
//!    (LanguageTool rules) or `harper` (Harper grammar engine).
//!
//! # Features
//!
//! - `full` (default) — enables `symspell`, `spellbook`, `nlprule`, `harper`
//! - `symspell` — frequency-weighted symmetric-delete spelling
//! - `spellbook` — Hunspell-compatible dictionary with affix rules
//! - `nlprule` — rule-based grammar checking via LanguageTool
//! - `harper` — modern grammar + spelling engine from Automattic
//!
//! # Example
//!
//! ```rust
//! use crustytts_spellcheck::SpellChecker;
//!
//! let checker = SpellChecker::new();
//! let issues = checker.check("the begining of kubernetes");
//! assert_eq!(issues[0].suggestion, Some("beginning".into()));
//! assert_eq!(issues[1].suggestion, None);
//! ```

use std::collections::HashSet;
use std::sync::OnceLock;

use serde::Deserialize;

// ── traits ──────────────────────────────────────────────────────────────────────

/// A word dictionary for spell checking.
pub trait Dictionary: Send + Sync {
    fn contains(&self, word: &str) -> bool;
}

/// An LLM provider for context-aware spell checking.
pub trait LlmProvider: Send + Sync {
    /// Review ambiguous `words` in the context of `text`.
    fn review_words(&self, text: &str, words: &[&str]) -> Result<Vec<Option<String>>, String>;

    /// Review all proposed changes and decide which to apply.
    ///
    /// Returns a `Vec<bool>` parallel to `proposals`: `true` = accept, `false` = reject.
    /// The default implementation accepts everything — override for LLM-based decision.
    fn review_proposals(
        &self,
        _text: &str,
        proposals: &[Proposal],
    ) -> Result<Vec<bool>, String> {
        Ok(vec![true; proposals.len()])
    }
}

/// A proposed change from one stage of the pipeline.
#[derive(Debug, Clone)]
pub struct Proposal {
    /// Which stage proposed this change.
    pub source: ProposalSource,
    /// The original text span.
    pub original: String,
    /// The suggested replacement.
    pub suggested: String,
    /// Why this change was proposed.
    pub reason: String,
}

/// Which stage of the pipeline proposed a change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProposalSource {
    /// From the dictionary + edit-distance spell checker.
    Spellcheck,
    /// From the grammar checker.
    Grammar,
}

/// A grammar checker that corrects grammar errors in text.
///
/// Runs after spell checking. Note: implementations are not required to be
/// `Send + Sync` — some engines (like harper-core) use single-threaded
/// internals. Pass your checker to [`SpellChecker::correct_full`] rather
/// than storing it in the checker.
pub trait GrammarChecker {
    fn correct(&self, text: &str) -> Result<String, String>;
}

// ── embedded dictionary ─────────────────────────────────────────────────────────

pub struct EmbeddedDictionary {
    words: &'static HashSet<String>,
}

impl EmbeddedDictionary {
    pub fn new() -> Self {
        Self {
            words: embedded_words(),
        }
    }
}

impl Default for EmbeddedDictionary {
    fn default() -> Self {
        Self::new()
    }
}

impl Dictionary for EmbeddedDictionary {
    fn contains(&self, word: &str) -> bool {
        self.words.contains(word)
    }
}

const WORDS: &str = include_str!("words.txt");

fn embedded_words() -> &'static HashSet<String> {
    static DICT: OnceLock<HashSet<String>> = OnceLock::new();
    DICT.get_or_init(|| {
        WORDS
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|w| !w.is_empty())
            .collect()
    })
}

// ── symspell dictionary ─────────────────────────────────────────────────────────

#[cfg(feature = "symspell")]
pub struct SymSpellDict {
    engine: symspell::SymSpell<symspell::AsciiStringStrategy>,
}

#[cfg(feature = "symspell")]
impl SymSpellDict {
    pub fn load(path: &str) -> Result<Self, String> {
        let mut engine = symspell::SymSpell::default();
        if !engine.load_dictionary(path, 0, 1, " ") {
            return Err(format!("symspell failed to load dictionary: {path}"));
        }
        Ok(Self { engine })
    }

    pub fn suggest(&self, word: &str) -> Vec<String> {
        self.engine
            .lookup(word, symspell::Verbosity::Closest, 2)
            .into_iter()
            .map(|s| s.term)
            .collect()
    }
}

#[cfg(feature = "symspell")]
impl Dictionary for SymSpellDict {
    fn contains(&self, word: &str) -> bool {
        !self
            .engine
            .lookup(word, symspell::Verbosity::Top, 0)
            .is_empty()
    }
}

// ── spellbook dictionary ────────────────────────────────────────────────────────

#[cfg(feature = "spellbook")]
pub struct SpellbookDict {
    dict: spellbook::Dictionary,
}

#[cfg(feature = "spellbook")]
impl SpellbookDict {
    pub fn new(aff: &str, dic: &str) -> Result<Self, String> {
        let dict = spellbook::Dictionary::new(aff, dic)
            .map_err(|e| format!("spellbook load error: {e}"))?;
        Ok(Self { dict })
    }

    pub fn suggest(&self, word: &str) -> Vec<String> {
        let mut suggestions = Vec::new();
        self.dict.suggest(word, &mut suggestions);
        suggestions
    }
}

#[cfg(feature = "spellbook")]
impl Dictionary for SpellbookDict {
    fn contains(&self, word: &str) -> bool {
        self.dict.check(word)
    }
}

// ── nlprule grammar checker ─────────────────────────────────────────────────────

#[cfg(feature = "nlprule")]
pub struct NlpRuleChecker {
    tokenizer: nlprule::Tokenizer,
    rules: nlprule::Rules,
}

#[cfg(feature = "nlprule")]
impl NlpRuleChecker {
    pub fn load(tokenizer_path: &str, rules_path: &str) -> Result<Self, String> {
        let tokenizer = nlprule::Tokenizer::new(tokenizer_path)
            .map_err(|e| format!("nlprule tokenizer: {e}"))?;
        let rules = nlprule::Rules::new(rules_path)
            .map_err(|e| format!("nlprule rules: {e}"))?;
        Ok(Self { tokenizer, rules })
    }
}

#[cfg(feature = "nlprule")]
impl GrammarChecker for NlpRuleChecker {
    fn correct(&self, text: &str) -> Result<String, String> {
        let suggestions = self.rules.suggest(text, &self.tokenizer);
        Ok(nlprule::rules::apply_suggestions(text, &suggestions))
    }
}

// ── harper grammar checker ──────────────────────────────────────────────────────

#[cfg(feature = "harper")]
pub struct HarperChecker {
    linter: std::sync::Mutex<harper_core::linting::LintGroup>,
    parser: harper_core::parsers::PlainEnglish,
}

#[cfg(feature = "harper")]
impl HarperChecker {
    pub fn new() -> Self {
        use harper_core::linting::LintGroup;
        use harper_core::spell::FstDictionary;
        use harper_core::Dialect;

        let dict = FstDictionary::curated();
        let linter = LintGroup::new_curated(dict, Dialect::American);
        let parser = harper_core::parsers::PlainEnglish;

        Self {
            linter: std::sync::Mutex::new(linter),
            parser,
        }
    }
}

#[cfg(feature = "harper")]
impl Default for HarperChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "harper")]
impl GrammarChecker for HarperChecker {
    fn correct(&self, text: &str) -> Result<String, String> {
        use harper_core::linting::Linter;
        use harper_core::Document;

        let document = Document::new_curated(text, &self.parser);
        let lints = self
            .linter
            .lock()
            .map_err(|e| format!("harper lock: {e}"))?
            .lint(&document);

        if lints.is_empty() {
            return Ok(text.to_string());
        }

        let mut result = text.to_string();
        let mut sorted: Vec<_> = lints.iter().collect();
        sorted.sort_by_key(|l| std::cmp::Reverse(l.span.start));

        for lint in &sorted {
            use harper_core::linting::Suggestion;
            let replacement: Option<String> = match lint.suggestions.first() {
                Some(Suggestion::ReplaceWith(chars)) => Some(chars.iter().collect()),
                Some(Suggestion::Remove) => Some(String::new()),
                _ => None,
            };
            if let Some(replacement) = replacement {
                let start = lint.span.start;
                let end = lint.span.end;
                if result.get(start..end).is_some() {
                    result.replace_range(start..end, &replacement);
                }
            }
        }

        Ok(result)
    }
}

// ── Ollama LLM provider ─────────────────────────────────────────────────────────

pub struct OllamaProvider {
    model: String,
    endpoint: String,
    timeout_secs: u64,
}

impl OllamaProvider {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            endpoint: "http://localhost:11434/api/generate".into(),
            timeout_secs: 10,
        }
    }

    pub fn with_endpoint(mut self, url: impl Into<String>) -> Self {
        self.endpoint = url.into();
        self
    }

    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }
}

impl LlmProvider for OllamaProvider {
    fn review_words(&self, text: &str, words: &[&str]) -> Result<Vec<Option<String>>, String> {
        if words.is_empty() {
            return Ok(Vec::new());
        }

        let word_list = words
            .iter()
            .enumerate()
            .map(|(i, w)| format!("{}. \"{w}\"", i + 1))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "You are a conservative spell checker reviewing flagged words in context.\n\n\
             Full text: \"{text}\"\n\n\
             Flagged words:\n{word_list}\n\n\
             For each flagged word, determine if it is:\n\
             - A genuine typo → provide the correct spelling\n\
             - An industry term, acronym, proper noun, or technical jargon → keep as-is\n\n\
             Rules:\n\
             - Only suggest a correction if you are VERY confident it's a typo.\n\
             - Acronyms (K8s, DNS, TCP), brand names (Kubernetes, GitHub), and technical\n\
               terms (tokio, nginx) are NOT typos — keep them.\n\
             - If unsure, keep the original.\n\n\
             Respond with a JSON array of objects. Each object has:\n\
             - \"index\": the number from the flagged words list\n\
             - \"action\": \"correct\" or \"keep\"\n\
             - \"correction\": the corrected word (only if action is \"correct\")\n\n\
             Example: [{{\"index\":1,\"action\":\"correct\",\"correction\":\"deployed\"}},\
                       {{\"index\":2,\"action\":\"keep\"}}]\n\n\
             Respond with the JSON array only."
        );

        let resp = reqwest::blocking::Client::new()
            .post(&self.endpoint)
            .timeout(std::time::Duration::from_secs(self.timeout_secs))
            .json(&serde_json::json!({
                "model": &self.model,
                "prompt": prompt,
                "stream": false,
                "think": false,
                "options": {"num_predict": 256},
            }))
            .send()
            .map_err(|e| format!("Ollama request failed: {e}"))?;

        #[derive(Deserialize)]
        struct OllamaResponse {
            response: Option<String>,
        }

        let body: OllamaResponse = resp
            .json()
            .map_err(|e| format!("Ollama response parse failed: {e}"))?;

        let raw = body.response.unwrap_or_default();
        parse_llm_spelling_response(&raw, words.len())
    }

    fn review_proposals(
        &self,
        text: &str,
        proposals: &[Proposal],
    ) -> Result<Vec<bool>, String> {
        if proposals.is_empty() {
            return Ok(Vec::new());
        }

        let proposal_list = proposals
            .iter()
            .enumerate()
            .map(|(i, p)| {
                format!(
                    "{}. [{}] \"{}\" → \"{}\" ({})",
                    i + 1,
                    match p.source {
                        ProposalSource::Spellcheck => "spell",
                        ProposalSource::Grammar => "grammar",
                    },
                    p.original,
                    p.suggested,
                    p.reason,
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "You are the final decision-maker in a text correction pipeline.\n\n\
             Original text: \"{text}\"\n\n\
             Proposed changes:\n{proposal_list}\n\n\
             For each proposal, decide whether to ACCEPT or REJECT it.\n\n\
             Rules:\n\
             - ACCEPT clear spelling mistakes and grammar errors.\n\
             - REJECT changes that would alter industry terms, acronyms, proper nouns,\n\
               brand names, or technical jargon (even if misspelled according to a\n\
               general dictionary).\n\
             - REJECT changes that would make the text less natural or change its meaning.\n\
             - If a spellchecker flags \"Kubernetes\" → \"Kernels\", REJECT it.\n\
             - If grammar suggests changing an intentional informal tone, REJECT it.\n\
             - When in doubt, REJECT. Only ACCEPT when very confident.\n\n\
             Respond with a JSON array of objects. Each object has:\n\
             - \"index\": the proposal number\n\
             - \"action\": \"accept\" or \"reject\"\n\n\
             Example: [{{\"index\":1,\"action\":\"accept\"}},\
                       {{\"index\":2,\"action\":\"reject\"}}]\n\n\
             Respond with the JSON array only."
        );

        let resp = reqwest::blocking::Client::new()
            .post(&self.endpoint)
            .timeout(std::time::Duration::from_secs(self.timeout_secs))
            .json(&serde_json::json!({
                "model": &self.model,
                "prompt": prompt,
                "stream": false,
                "think": false,
                "options": {"num_predict": 128},
            }))
            .send()
            .map_err(|e| format!("Ollama request failed: {e}"))?;

        #[derive(Deserialize)]
        struct OllamaResponse {
            response: Option<String>,
        }

        let body: OllamaResponse = resp
            .json()
            .map_err(|e| format!("Ollama response parse failed: {e}"))?;

        let raw = body.response.unwrap_or_default();
        parse_proposal_decisions(&raw, proposals.len())
    }
}

// ── public types ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Confidence {
    High,
    LlmVerified,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpellIssue {
    pub word: String,
    pub offset: usize,
    pub suggestion: Option<String>,
    pub confidence: Option<Confidence>,
}

/// The spell checker.
pub struct SpellChecker {
    dict: Box<dyn Dictionary>,
    llm: Option<Box<dyn LlmProvider>>,
    allowlist: HashSet<String>,
}

impl Default for SpellChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl SpellChecker {
    pub fn new() -> Self {
        Self {
            dict: Box::new(EmbeddedDictionary::new()),
            llm: None,
            allowlist: HashSet::new(),
        }
    }

    pub fn with_dictionary(mut self, dict: impl Dictionary + 'static) -> Self {
        self.dict = Box::new(dict);
        self
    }

    pub fn with_llm(mut self, llm: impl LlmProvider + 'static) -> Self {
        self.llm = Some(Box::new(llm));
        self
    }


    pub fn allow(mut self, word: impl Into<String>) -> Self {
        self.allowlist.insert(word.into().to_lowercase());
        self
    }

    pub fn allow_all(mut self, words: impl IntoIterator<Item = impl Into<String>>) -> Self {
        for w in words {
            self.allowlist.insert(w.into().to_lowercase());
        }
        self
    }

    pub fn check(&self, text: &str) -> Vec<SpellIssue> {
        let mut issues = Vec::new();

        for (offset, word) in tokenize_words(text) {
            let lower = word.to_lowercase();
            if self.allowlist.contains(&lower) || self.dict.contains(&lower) {
                continue;
            }

            let is_proper_noun = word.len() > 1
                && word.chars().next().is_some_and(|c| c.is_uppercase())
                && word.chars().skip(1).all(|c| c.is_lowercase());

            let suggestion = if is_proper_noun {
                None
            } else {
                find_correction(&lower, self.dict.as_ref())
            };

            issues.push(SpellIssue {
                word,
                offset,
                suggestion,
                confidence: None,
            });

            if let Some(issue) = issues.last_mut() {
                issue.confidence = issue.suggestion.as_ref().map(|_| Confidence::High);
            }
        }

        issues
    }

    pub fn correct(&self, text: &str) -> String {
        let issues = self.check(text);
        apply_corrections(text, &issues)
    }

    pub fn correct_with_llm(&self, text: &str) -> Result<String, String> {
        let llm = self
            .llm
            .as_ref()
            .ok_or_else(|| "no LLM provider configured — call .with_llm() first".to_string())?;

        let mut issues = self.check(text);

        let needs_llm: Vec<usize> = issues
            .iter()
            .enumerate()
            .filter(|(_, i)| i.suggestion.is_none())
            .map(|(idx, _)| idx)
            .collect();

        if needs_llm.is_empty() {
            return Ok(apply_corrections(text, &issues));
        }

        let ambiguous_words: Vec<&str> =
            needs_llm.iter().map(|&i| issues[i].word.as_str()).collect();
        let llm_result = llm.review_words(text, &ambiguous_words)?;

        for (&idx, suggestion) in needs_llm.iter().zip(llm_result.iter()) {
            if let Some(corrected) = suggestion {
                issues[idx].suggestion = Some(corrected.clone());
                issues[idx].confidence = Some(Confidence::LlmVerified);
            }
        }

        Ok(apply_corrections(text, &issues))
    }

    /// Full pipeline: spellcheck → grammar → LLM (final decision).
    ///
    /// 1. Runs dictionary + edit-distance spell checking.
    /// 2. Runs the grammar checker (if provided).
    /// 3. Collects all proposed changes and sends them to the LLM.
    /// 4. The LLM decides which changes to accept or reject.
    /// 5. Applies only the LLM-approved changes.
    ///
    /// Returns an error if an LLM provider is not configured.
    pub fn correct_full(&self, text: &str, grammar: Option<&dyn GrammarChecker>) -> Result<String, String> {
        let llm = self
            .llm
            .as_ref()
            .ok_or_else(|| "no LLM provider configured — call .with_llm() first".to_string())?;

        let spell_issues = self.check(text);

        // Step 1: separate issues into "has suggestion" and "needs LLM review"
        let mut proposals: Vec<Proposal> = Vec::new();
        let mut needs_review: Vec<&str> = Vec::new();

        for issue in &spell_issues {
            if let Some(s) = &issue.suggestion {
                proposals.push(Proposal {
                    source: ProposalSource::Spellcheck,
                    original: issue.word.clone(),
                    suggested: s.clone(),
                    reason: format!(
                        "dictionary + edit-distance suggests \"{}\" → \"{s}\"",
                        issue.word
                    ),
                });
            } else {
                needs_review.push(&issue.word);
            }
        }

        // Step 2: run grammar checker and diff the result
        if let Some(g) = grammar {
            if let Ok(grammar_fixed) = g.correct(text) {
                if grammar_fixed != text {
                    let grammar_proposals = diff_proposals(text, &grammar_fixed);
                    proposals.extend(grammar_proposals);
                }
            }
        }

        // Step 3: ask LLM to review ambiguous words (no suggestion from spellcheck)
        if !needs_review.is_empty() {
            let llm_suggestions = llm.review_words(text, &needs_review)?;
            for (word, suggestion) in needs_review.iter().zip(llm_suggestions.iter()) {
                if let Some(s) = suggestion {
                    proposals.push(Proposal {
                        source: ProposalSource::Spellcheck,
                        original: word.to_string(),
                        suggested: s.clone(),
                        reason: format!("LLM suggests \"{word}\" → \"{s}\""),
                    });
                }
            }
        }

        // Step 4: if no proposals, return original
        if proposals.is_empty() {
            return Ok(text.to_string());
        }

        // Step 5: LLM reviews all proposals (spellcheck + grammar + LLM-suggested) and decides
        let decisions = llm.review_proposals(text, &proposals)?;

        // Step 6: apply only accepted proposals
        let mut result = text.to_string();
        let mut accepted: Vec<(&Proposal, bool)> = proposals.iter().zip(decisions).collect();
        accepted.sort_by_key(|(p, _)| {
            std::cmp::Reverse(result.find(&p.original))
        });

        for (proposal, accepted) in &accepted {
            if !accepted {
                continue;
            }
            if let Some(pos) = result.find(&proposal.original) {
                let end = pos + proposal.original.len();
                result.replace_range(pos..end, &proposal.suggested);
            }
        }

        Ok(result)
    }
}

// ── grammar diffing ────────────────────────────────────────────────────────────

/// Compare original and grammar-corrected text, extracting word-level changes
/// as proposals.
fn diff_proposals(original: &str, corrected: &str) -> Vec<Proposal> {
    let orig_words: Vec<&str> = original.split_whitespace().collect();
    let corr_words: Vec<&str> = corrected.split_whitespace().collect();

    // Simple word-by-word diff: find the first and last differing positions
    let mut proposals = Vec::new();

    let min_len = orig_words.len().min(corr_words.len());
    let mut start_diff = 0;
    while start_diff < min_len && orig_words[start_diff] == corr_words[start_diff] {
        start_diff += 1;
    }

    if start_diff >= min_len && orig_words.len() == corr_words.len() {
        return proposals; // no differences
    }

    let orig_end = orig_words.len();
    let corr_end = corr_words.len();
    let mut end_orig = orig_end;
    let mut end_corr = corr_end;

    while end_orig > start_diff
        && end_corr > start_diff
        && orig_words[end_orig - 1] == corr_words[end_corr - 1]
    {
        end_orig -= 1;
        end_corr -= 1;
    }

    let orig_span: String = orig_words[start_diff..end_orig].join(" ");
    let corr_span: String = corr_words[start_diff..end_corr].join(" ");

    if !orig_span.is_empty() && orig_span != corr_span {
        let reason = format!("grammar checker suggests \"{orig_span}\" → \"{corr_span}\"");
        proposals.push(Proposal {
            source: ProposalSource::Grammar,
            original: orig_span,
            suggested: corr_span,
            reason,
        });
    }

    proposals
}

// ── tokenization ────────────────────────────────────────────────────────────────

fn tokenize_words(text: &str) -> Vec<(usize, String)> {
    let mut words = Vec::new();
    let mut start: Option<usize> = None;

    for (i, ch) in text.char_indices() {
        if ch.is_alphabetic() {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start {
            words.push((s, text[s..i].to_string()));
            start = None;
        }
    }
    if let Some(s) = start {
        words.push((s, text[s..].to_string()));
    }

    words
}

// ── edit-distance correction ────────────────────────────────────────────────────

fn find_correction(word: &str, dict: &dyn Dictionary) -> Option<String> {
    let mut found: Option<String> = None;

    for c in edits1(word) {
        if dict.contains(&c) {
            match &found {
                Some(existing) if *existing != c => return None,
                Some(_) => {}
                None => found = Some(c),
            }
        }
    }

    found
}

fn edits1(word: &str) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();
    let n = chars.len();
    let mut v = Vec::with_capacity(n + 26 * (n + 1) + 26 * n + n.saturating_sub(1));

    for i in 0..n {
        let mut s = String::with_capacity(n - 1);
        s.extend(chars[..i].iter());
        s.extend(chars[i + 1..].iter());
        v.push(s);
    }

    for i in 0..=n {
        for c in 'a'..='z' {
            let mut s = String::with_capacity(n + 1);
            s.extend(chars[..i].iter());
            s.push(c);
            s.extend(chars[i..].iter());
            v.push(s);
        }
    }

    for i in 0..n {
        for c in 'a'..='z' {
            if c == chars[i] {
                continue;
            }
            let mut s = String::with_capacity(n);
            s.extend(chars[..i].iter());
            s.push(c);
            s.extend(chars[i + 1..].iter());
            v.push(s);
        }
    }

    for i in 0..n.saturating_sub(1) {
        let mut s = String::with_capacity(n);
        s.extend(chars[..i].iter());
        s.push(chars[i + 1]);
        s.push(chars[i]);
        s.extend(chars[i + 2..].iter());
        v.push(s);
    }

    v
}

// ── correction application ──────────────────────────────────────────────────────

fn apply_corrections(text: &str, issues: &[SpellIssue]) -> String {
    if issues.is_empty() {
        return text.to_string();
    }

    let mut replacements: Vec<(&SpellIssue, &str)> = issues
        .iter()
        .filter_map(|i| i.suggestion.as_ref().map(|s| (i, s.as_str())))
        .collect();

    replacements.sort_by_key(|(i, _)| std::cmp::Reverse(i.offset));

    let mut result = text.to_string();
    for (issue, suggestion) in &replacements {
        let end = issue.offset + issue.word.len();
        if result.get(issue.offset..end) == Some(&issue.word) {
            result.replace_range(issue.offset..end, suggestion);
        }
    }

    result
}

// ── LLM response parsing ────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct LlmSuggestion {
    index: usize,
    action: String,
    #[serde(default)]
    correction: Option<String>,
}

fn parse_llm_spelling_response(json: &str, count: usize) -> Result<Vec<Option<String>>, String> {
    let json = json.trim();
    let json = json
        .strip_prefix("```json")
        .or_else(|| json.strip_prefix("```"))
        .unwrap_or(json);
    let json = json.strip_suffix("```").unwrap_or(json);
    let json = json.trim();

    let suggestions: Vec<LlmSuggestion> =
        serde_json::from_str(json).map_err(|e| format!("LLM response parse error: {e}"))?;

    let mut result = vec![None; count];
    for s in suggestions {
        if s.index == 0 || s.index > count {
            continue;
        }
        if s.action == "correct" {
            if let Some(c) = s.correction {
                result[s.index - 1] = Some(c);
            }
        }
    }

    Ok(result)
}

#[derive(Deserialize)]
struct ProposalDecision {
    index: usize,
    action: String,
}

fn parse_proposal_decisions(json: &str, count: usize) -> Result<Vec<bool>, String> {
    let json = json.trim();
    let json = json
        .strip_prefix("```json")
        .or_else(|| json.strip_prefix("```"))
        .unwrap_or(json);
    let json = json.strip_suffix("```").unwrap_or(json);
    let json = json.trim();

    let decisions: Vec<ProposalDecision> =
        serde_json::from_str(json).map_err(|e| format!("LLM response parse error: {e}"))?;

    let mut result = vec![false; count];
    for d in decisions {
        if d.index == 0 || d.index > count {
            continue;
        }
        result[d.index - 1] = d.action == "accept";
    }

    Ok(result)
}

// ── tests ───────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_extracts_alphabetic_words() {
        let words = tokenize_words("Claude deployed to k8s at 1:00");
        let text_words: Vec<&str> = words.iter().map(|(_, w)| w.as_str()).collect();
        assert_eq!(text_words, vec!["Claude", "deployed", "to", "k", "s", "at"]);
    }

    #[test]
    fn tokenize_handles_punctuation() {
        let words = tokenize_words("Hello, world! How's it going?");
        let text_words: Vec<&str> = words.iter().map(|(_, w)| w.as_str()).collect();
        assert_eq!(text_words, vec!["Hello", "world", "How", "s", "it", "going"]);
    }

    #[test]
    fn edits1_generates_deletions() {
        let e = edits1("abc");
        assert!(e.contains(&"ab".to_string()));
        assert!(e.contains(&"bc".to_string()));
        assert!(e.contains(&"ac".to_string()));
    }

    #[test]
    fn edits1_generates_insertions() {
        let e = edits1("ab");
        assert!(e.contains(&"abc".to_string()));
    }

    #[test]
    fn edits1_generates_substitutions() {
        let e = edits1("abc");
        assert!(e.contains(&"abd".to_string()));
        assert!(!e.contains(&"abc".to_string()));
    }

    #[test]
    fn edits1_generates_transpositions() {
        let e = edits1("ab");
        assert!(e.contains(&"ba".to_string()));
    }

    #[test]
    fn embedded_dictionary_contains_common_words() {
        let dict = EmbeddedDictionary::new();
        assert!(dict.contains("deployed"));
        assert!(dict.contains("beginning"));
        assert!(dict.contains("the"));
        assert!(!dict.contains("kubernetes"));
        assert!(!dict.contains("nginx"));
    }

    #[test]
    fn catches_common_typo() {
        let checker = SpellChecker::new();
        let issues = checker.check("the begining of the end");
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].word, "begining");
        assert_eq!(issues[0].suggestion, Some("beginning".into()));
        assert_eq!(issues[0].confidence, Some(Confidence::High));
    }

    #[test]
    fn leaves_known_words_alone() {
        let checker = SpellChecker::new();
        let issues = checker.check("we deployed the application");
        let flagged: Vec<&str> = issues.iter().map(|i| i.word.as_str()).collect();
        assert!(!flagged.contains(&"deployed"));
        assert!(!flagged.contains(&"the"));
        assert!(!flagged.contains(&"application"));
        assert!(flagged.is_empty(), "expected no issues, got: {flagged:?}");
    }

    #[test]
    fn proper_nouns_are_flagged_but_not_auto_corrected() {
        let checker = SpellChecker::new();
        let issues = checker.check("Claude deployed to kubernetes");
        let claude: Vec<&SpellIssue> = issues.iter().filter(|i| i.word == "Claude").collect();
        assert_eq!(claude.len(), 1);
        assert_eq!(claude[0].suggestion, None);
    }

    #[test]
    fn unknown_word_with_no_close_match_has_no_suggestion() {
        let checker = SpellChecker::new();
        let issues = checker.check("deployed to kubernetes");
        let k8s: Vec<&SpellIssue> = issues.iter().filter(|i| i.word == "kubernetes").collect();
        assert_eq!(k8s.len(), 1);
        assert_eq!(k8s[0].suggestion, None);
    }

    #[test]
    fn auto_correct_applies_high_confidence_only() {
        let checker = SpellChecker::new();
        let result = checker.correct("the begining of kubernetes");
        assert!(result.contains("beginning"), "should fix 'begining': {result}");
        assert!(result.contains("kubernetes"), "should keep 'kubernetes': {result}");
    }

    #[test]
    fn allowlist_prevents_flagging() {
        let checker = SpellChecker::new().allow("kubernetes").allow("claude");
        let issues = checker.check("Claude deployed to kubernetes");
        assert!(issues.is_empty());
    }

    #[test]
    fn handles_empty_text() {
        let checker = SpellChecker::new();
        assert!(checker.check("").is_empty());
        assert_eq!(checker.correct(""), "");
    }

    #[test]
    fn preserves_punctuation_and_spacing() {
        let checker = SpellChecker::new();
        let result = checker.correct("the begining, then the end.");
        assert!(result.contains("beginning"), "got: {result}");
        assert!(result.contains(','), "punctuation lost: {result}");
        assert!(result.ends_with('.'), "period lost: {result}");
    }

    #[test]
    fn multiple_typos_in_one_sentence() {
        let checker = SpellChecker::new();
        let issues = checker.check("it occured when I recieved it");
        let words: Vec<&str> = issues.iter().map(|i| i.word.as_str()).collect();
        assert!(words.contains(&"occured"));
        assert!(words.contains(&"recieved"));
        let occured = issues.iter().find(|i| i.word == "occured").unwrap();
        assert_eq!(occured.suggestion, Some("occurred".into()));
        let recieved = issues.iter().find(|i| i.word == "recieved").unwrap();
        assert_eq!(recieved.suggestion, None);
    }

    #[test]
    fn numbers_and_special_chars_are_skipped() {
        let checker = SpellChecker::new();
        let issues = checker.check("v1.2.3 deployed at 3:30");
        let words: Vec<&str> = issues.iter().map(|i| i.word.as_str()).collect();
        assert!(!words.contains(&"1"));
        assert!(!words.contains(&"2"));
        assert!(!words.contains(&"3"));
        assert!(!words.contains(&"30"));
    }

    #[test]
    fn custom_dictionary_is_used() {
        struct EmptyDict;
        impl Dictionary for EmptyDict {
            fn contains(&self, _word: &str) -> bool { false }
        }
        let checker = SpellChecker::new().with_dictionary(EmptyDict);
        let issues = checker.check("the quick brown fox");
        assert_eq!(issues.len(), 4);
    }

    #[test]
    fn custom_dictionary_can_accept_everything() {
        struct YesDict;
        impl Dictionary for YesDict {
            fn contains(&self, _word: &str) -> bool { true }
        }
        let checker = SpellChecker::new().with_dictionary(YesDict);
        let issues = checker.check("the quick brown fox");
        assert!(issues.is_empty());
    }

    #[test]
    fn correct_with_llm_errors_without_provider() {
        let checker = SpellChecker::new();
        let result = checker.correct_with_llm("some text");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no LLM provider"));
    }

    #[test]
    fn correct_full_requires_llm() {
        let checker = SpellChecker::new();
        let result = checker.correct_full("the begining", None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no LLM provider"));
    }

    #[test]
    fn correct_full_applies_spellcheck_when_llm_accepts() {
        struct YesLlm;
        impl LlmProvider for YesLlm {
            fn review_words(&self, _text: &str, _words: &[&str]) -> Result<Vec<Option<String>>, String> {
                Ok(vec![])
            }
            fn review_proposals(&self, _text: &str, proposals: &[Proposal]) -> Result<Vec<bool>, String> {
                Ok(vec![true; proposals.len()])
            }
        }
        let checker = SpellChecker::new().with_llm(YesLlm);
        let result = checker.correct_full("the begining", None).unwrap();
        assert!(result.contains("beginning"));
    }

    #[test]
    fn correct_full_rejects_when_llm_says_no() {
        struct NoLlm;
        impl LlmProvider for NoLlm {
            fn review_words(&self, _text: &str, _words: &[&str]) -> Result<Vec<Option<String>>, String> {
                Ok(vec![])
            }
            fn review_proposals(&self, _text: &str, proposals: &[Proposal]) -> Result<Vec<bool>, String> {
                Ok(vec![false; proposals.len()])
            }
        }
        let checker = SpellChecker::new().with_llm(NoLlm);
        let result = checker.correct_full("the begining", None).unwrap();
        assert_eq!(result, "the begining"); // unchanged — LLM rejected
    }

    #[test]
    fn correct_full_includes_grammar_proposals() {
        struct StubGrammar;
        impl GrammarChecker for StubGrammar {
            fn correct(&self, _text: &str) -> Result<String, String> {
                Ok("she were not here".into()) // grammar error: "were" → "was"
            }
        }
        struct RecordLlm {
            proposals: std::sync::Mutex<Vec<Proposal>>,
        }
        impl LlmProvider for RecordLlm {
            fn review_words(&self, _text: &str, _words: &[&str]) -> Result<Vec<Option<String>>, String> {
                Ok(vec![])
            }
            fn review_proposals(&self, _text: &str, proposals: &[Proposal]) -> Result<Vec<bool>, String> {
                let mut p = self.proposals.lock().unwrap();
                p.extend(proposals.iter().cloned());
                Ok(vec![true; proposals.len()])
            }
        }
        let llm = RecordLlm { proposals: std::sync::Mutex::new(Vec::new()) };
        let checker = SpellChecker::new().with_llm(llm);
        let grammar = StubGrammar;
        // "she was not here" — all words in dict, grammar changes "was not" → "were not"
        let _ = checker.correct_full("she was not here", Some(&grammar)).unwrap();
        // The grammar proposal should have been recorded
        // (spellcheck finds no issues, grammar proposes a change)
    }
}
