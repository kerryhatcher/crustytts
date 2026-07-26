//! Chat-speak abbreviation expansion and acronym capitalization.
//!
//! A deterministic, zero-dependency normalizer that expands common chat
//! abbreviations (`ur` → `your`, `dev` → `development`) and capitalizes
//! known tech acronyms (`gui` → `GUI`, `api` → `API`).
//!
//! Implements [`crustytts_core::ProofingStage`] so it can be plugged into a
//! proofreading pipeline.
//!
//! # Example
//!
//! ```rust
//! use crustytts_chatspeak::ChatSpeakNormalizer;
//! use crustytts_core::ProofingStage;
//!
//! let cs = ChatSpeakNormalizer::new();
//! assert_eq!(cs.proof("ur dev work on gui"), "your development work on GUI");
//! ```

use crustytts_core::ProofingStage;
use std::collections::HashMap;

/// Expands chat abbreviations and capitalizes known acronyms.
///
/// The normalizer is case-insensitive on input but preserves the casing of
/// the expansion. Acronyms are always uppercased.
pub struct ChatSpeakNormalizer {
    abbreviations: HashMap<&'static str, &'static str>,
    acronyms: HashMap<&'static str, &'static str>,
}

impl ChatSpeakNormalizer {
    /// Create a new normalizer with the built-in abbreviation and acronym tables.
    pub fn new() -> Self {
        Self {
            abbreviations: default_abbreviations(),
            acronyms: default_acronyms(),
        }
    }

    /// Add a custom abbreviation mapping.
    pub fn with_abbreviation(mut self, short: &'static str, long: &'static str) -> Self {
        self.abbreviations.insert(short, long);
        self
    }

    /// Add a custom acronym that should always be uppercased.
    ///
    /// The acronym is stored lowercase for matching and the display form
    /// is used as the replacement.
    pub fn with_acronym(mut self, lowercase: &'static str, display: &'static str) -> Self {
        self.acronyms.insert(lowercase, display);
        self
    }
}

impl Default for ChatSpeakNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl ProofingStage for ChatSpeakNormalizer {
    fn proof(&self, text: &str) -> String {
        let mut result = String::with_capacity(text.len() + 32);
        let mut word_start: Option<usize> = None;
        let chars: Vec<char> = text.chars().collect();

        for (i, ch) in chars.iter().enumerate() {
            if ch.is_alphanumeric() || *ch == '\'' {
                if word_start.is_none() {
                    word_start = Some(i);
                }
            } else {
                if let Some(start) = word_start {
                    let word: String = chars[start..i].iter().collect();
                    result.push_str(&self.expand_word(&word));
                    word_start = None;
                }
                result.push(*ch);
            }
        }

        if let Some(start) = word_start {
            let word: String = chars[start..].iter().collect();
            result.push_str(&self.expand_word(&word));
        }

        result
    }
}

impl ChatSpeakNormalizer {
    /// Look up a single word: try abbreviation expansion, then acronym
    /// capitalization, then return as-is.
    fn expand_word(&self, word: &str) -> String {
        let lower = word.to_lowercase();

        // 1. Exact abbreviation match (case-insensitive)
        if let Some(expanded) = self.abbreviations.get(lower.as_str()) {
            // Preserve capitalization: if input was all-caps, output all-caps
            if word.chars().all(|c| c.is_uppercase()) {
                return expanded.to_uppercase();
            }
            // If input was title-case, title-case the output
            if word
                .chars()
                .next()
                .is_some_and(|c| c.is_uppercase())
            {
                return titlecase_first(expanded);
            }
            return expanded.to_string();
        }

        // 2. Known acronym that should be uppercased
        if let Some(acronym) = self.acronyms.get(lower.as_str()) {
            return acronym.to_string();
        }

        // 3. Pass through unchanged
        word.to_string()
    }
}

fn titlecase_first(s: &str) -> String {
    let mut chars: Vec<char> = s.chars().collect();
    if let Some(first) = chars.first_mut() {
        *first = first.to_uppercase().next().unwrap_or(*first);
    }
    chars.into_iter().collect()
}

// ── abbreviation tables ────────────────────────────────────────────────────────

fn default_abbreviations() -> HashMap<&'static str, &'static str> {
    let mut m = HashMap::new();

    // ── pronouns / common ──
    m.insert("ur", "your");
    m.insert("u", "you");
    m.insert("r", "are");
    m.insert("y", "why");
    m.insert("k", "okay");
    m.insert("thx", "thanks");
    m.insert("pls", "please");
    m.insert("plz", "please");
    m.insert("sry", "sorry");
    m.insert("idk", "I don't know");
    m.insert("imo", "in my opinion");
    m.insert("imho", "in my humble opinion");
    m.insert("tbh", "to be honest");
    m.insert("btw", "by the way");
    m.insert("afaik", "as far as I know");
    m.insert("iirc", "if I recall correctly");
    m.insert("fwiw", "for what it's worth");
    m.insert("ttyl", "talk to you later");
    m.insert("brb", "be right back");
    m.insert("gtg", "got to go");
    m.insert("g2g", "got to go");
    m.insert("omw", "on my way");
    m.insert("np", "no problem");
    m.insert("nvm", "never mind");
    m.insert("yw", "you're welcome");
    m.insert("ty", "thank you");
    m.insert("tysm", "thank you so much");
    m.insert("lol", "laughing out loud");
    m.insert("lmao", "laughing my ass off");
    m.insert("rofl", "rolling on the floor laughing");
    m.insert("omg", "oh my god");
    m.insert("wtf", "what the fuck");
    m.insert("ikr", "I know right");
    m.insert("fr", "for real");
    m.insert("ngl", "not gonna lie");
    m.insert("rn", "right now");
    m.insert("afk", "away from keyboard");
    m.insert("wfh", "working from home");
    m.insert("fyi", "for your information");
    m.insert("asap", "as soon as possible");
    m.insert("eta", "estimated time of arrival");
    m.insert("tba", "to be announced");
    m.insert("tbd", "to be determined");
    m.insert("wip", "work in progress");
    m.insert("poc", "proof of concept");
    m.insert("mvp", "minimum viable product");

    // ── dev / tech abbreviations ──
    m.insert("dev", "development");
    m.insert("impl", "implementation");
    m.insert("impls", "implementations");
    m.insert("cfg", "configuration");
    m.insert("config", "configuration");
    m.insert("env", "environment");
    m.insert("envs", "environments");
    m.insert("vars", "variables");
    m.insert("func", "function");
    m.insert("funcs", "functions");
    m.insert("var", "variable");
    m.insert("arg", "argument");
    m.insert("args", "arguments");
    m.insert("param", "parameter");
    m.insert("params", "parameters");
    m.insert("lib", "library");
    m.insert("libs", "libraries");
    m.insert("pkg", "package");
    m.insert("pkgs", "packages");
    m.insert("dep", "dependency");
    m.insert("deps", "dependencies");
    m.insert("repo", "repository");
    m.insert("repos", "repositories");
    m.insert("db", "database");
    m.insert("dbs", "databases");
    m.insert("svc", "service");
    m.insert("svcs", "services");
    m.insert("auth", "authentication");
    m.insert("authn", "authentication");
    m.insert("authz", "authorization");
    m.insert("perf", "performance");
    m.insert("opt", "optimization");
    m.insert("opts", "optimizations");
    m.insert("init", "initialization");
    m.insert("sync", "synchronization");
    m.insert("async", "asynchronous");
    m.insert("tmp", "temporary");
    m.insert("temp", "temporary");
    m.insert("dir", "directory");
    m.insert("dirs", "directories");
    m.insert("src", "source");
    m.insert("dest", "destination");
    m.insert("msg", "message");
    m.insert("msgs", "messages");
    m.insert("err", "error");
    m.insert("errs", "errors");
    m.insert("req", "request");
    m.insert("reqs", "requests");
    m.insert("resp", "response");
    m.insert("resps", "responses");
    m.insert("addr", "address");
    m.insert("addrs", "addresses");
    m.insert("mem", "memory");
    m.insert("cpu", "CPU");
    m.insert("gpu", "GPU");
    m.insert("ram", "RAM");
    m.insert("hdd", "hard drive");
    m.insert("ssd", "solid state drive");
    m.insert("os", "operating system");
    m.insert("vm", "virtual machine");
    m.insert("vms", "virtual machines");
    m.insert("k8s", "Kubernetes");
    m.insert("gh", "GitHub");
    m.insert("pr", "pull request");
    m.insert("prs", "pull requests");
    m.insert("ci", "continuous integration");
    m.insert("cd", "continuous deployment");
    m.insert("cicd", "CI/CD pipeline");
    m.insert("dx", "developer experience");
    m.insert("ux", "user experience");
    m.insert("ui", "user interface");
    m.insert("dx", "developer experience");

    // ── contractions (without apostrophe) ──
    m.insert("dont", "don't");
    m.insert("cant", "can't");
    m.insert("wont", "won't");
    m.insert("isnt", "isn't");
    m.insert("arent", "aren't");
    m.insert("wasnt", "wasn't");
    m.insert("werent", "weren't");
    m.insert("hasnt", "hasn't");
    m.insert("havent", "haven't");
    m.insert("hadnt", "hadn't");
    m.insert("doesnt", "doesn't");
    m.insert("didnt", "didn't");
    m.insert("shouldnt", "shouldn't");
    m.insert("wouldnt", "wouldn't");
    m.insert("couldnt", "couldn't");
    m.insert("mightnt", "mightn't");
    m.insert("mustnt", "mustn't");
    m.insert("neednt", "needn't");
    m.insert("theyll", "they'll");
    m.insert("theyd", "they'd");
    m.insert("theyve", "they've");
    m.insert("theyre", "they're");
    m.insert("well", "we'll");
    m.insert("wed", "we'd");
    m.insert("weve", "we've");
    m.insert("were", "we're");
    m.insert("youll", "you'll");
    m.insert("youd", "you'd");
    m.insert("youve", "you've");
    m.insert("youre", "you're");
    m.insert("ill", "I'll");
    m.insert("id", "I'd");
    m.insert("ive", "I've");
    m.insert("im", "I'm");
    m.insert("its", "it's");
    m.insert("thats", "that's");
    m.insert("whats", "what's");
    m.insert("whos", "who's");
    m.insert("wheres", "where's");
    m.insert("whens", "when's");
    m.insert("whys", "why's");
    m.insert("hows", "how's");
    m.insert("heres", "here's");
    m.insert("theres", "there's");
    m.insert("lets", "let's");

    m
}

fn default_acronyms() -> HashMap<&'static str, &'static str> {
    let mut m = HashMap::new();

    // ── tech acronyms (always uppercase) ──
    m.insert("gui", "GUI");
    m.insert("tui", "TUI");
    m.insert("cli", "CLI");
    m.insert("api", "API");
    m.insert("sdk", "SDK");
    m.insert("ide", "IDE");
    m.insert("orm", "ORM");
    m.insert("cdn", "CDN");
    m.insert("dns", "DNS");
    m.insert("tcp", "TCP");
    m.insert("udp", "UDP");
    m.insert("ip", "IP");
    m.insert("http", "HTTP");
    m.insert("https", "HTTPS");
    m.insert("ssl", "SSL");
    m.insert("tls", "TLS");
    m.insert("ssh", "SSH");
    m.insert("ftp", "FTP");
    m.insert("smtp", "SMTP");
    m.insert("imap", "IMAP");
    m.insert("pop3", "POP3");
    m.insert("sql", "SQL");
    m.insert("nosql", "NoSQL");
    m.insert("json", "JSON");
    m.insert("yaml", "YAML");
    m.insert("toml", "TOML");
    m.insert("xml", "XML");
    m.insert("html", "HTML");
    m.insert("css", "CSS");
    m.insert("scss", "SCSS");
    m.insert("sass", "Sass");
    m.insert("js", "JavaScript");
    m.insert("ts", "TypeScript");
    m.insert("jsx", "JSX");
    m.insert("tsx", "TSX");
    m.insert("wasm", "WASM");
    m.insert("jwt", "JWT");
    m.insert("oauth", "OAuth");
    m.insert("oauth2", "OAuth2");
    m.insert("cors", "CORS");
    m.insert("csrf", "CSRF");
    m.insert("xss", "XSS");
    m.insert("dom", "DOM");
    m.insert("svg", "SVG");
    m.insert("png", "PNG");
    m.insert("jpg", "JPEG");
    m.insert("jpeg", "JPEG");
    m.insert("gif", "GIF");
    m.insert("mp4", "MP4");
    m.insert("mp3", "MP3");
    m.insert("wav", "WAV");
    m.insert("pdf", "PDF");
    m.insert("csv", "CSV");
    m.insert("utf8", "UTF-8");
    m.insert("utf-8", "UTF-8");
    m.insert("ascii", "ASCII");
    m.insert("unicode", "Unicode");
    m.insert("posix", "POSIX");
    m.insert("linux", "Linux");
    m.insert("unix", "Unix");
    m.insert("macos", "macOS");
    m.insert("ios", "iOS");
    m.insert("ipados", "iPadOS");
    m.insert("android", "Android");
    m.insert("windows", "Windows");
    m.insert("aws", "AWS");
    m.insert("gcp", "GCP");
    m.insert("azure", "Azure");
    m.insert("vpc", "VPC");
    m.insert("ec2", "EC2");
    m.insert("s3", "S3");
    m.insert("rds", "RDS");
    m.insert("lambda", "Lambda");
    m.insert("docker", "Docker");
    m.insert("kubernetes", "Kubernetes");
    m.insert("nginx", "nginx");
    m.insert("postgres", "Postgres");
    m.insert("postgresql", "PostgreSQL");
    m.insert("mysql", "MySQL");
    m.insert("mariadb", "MariaDB");
    m.insert("mongodb", "MongoDB");
    m.insert("redis", "Redis");
    m.insert("sqlite", "SQLite");
    m.insert("elasticsearch", "Elasticsearch");
    m.insert("grafana", "Grafana");
    m.insert("prometheus", "Prometheus");
    m.insert("terraform", "Terraform");
    m.insert("ansible", "Ansible");
    m.insert("pulumi", "Pulumi");
    m.insert("git", "Git");
    m.insert("github", "GitHub");
    m.insert("gitlab", "GitLab");
    m.insert("bitbucket", "Bitbucket");
    m.insert("vscode", "VS Code");
    m.insert("intellij", "IntelliJ");
    m.insert("webstorm", "WebStorm");
    m.insert("rust", "Rust");
    m.insert("golang", "Go");
    m.insert("python", "Python");
    m.insert("javascript", "JavaScript");
    m.insert("typescript", "TypeScript");
    m.insert("react", "React");
    m.insert("vue", "Vue");
    m.insert("svelte", "Svelte");
    m.insert("angular", "Angular");
    m.insert("nextjs", "Next.js");
    m.insert("next.js", "Next.js");
    m.insert("nuxt", "Nuxt");
    m.insert("nodejs", "Node.js");
    m.insert("node.js", "Node.js");
    m.insert("deno", "Deno");
    m.insert("bun", "Bun");
    m.insert("webpack", "Webpack");
    m.insert("vite", "Vite");
    m.insert("esbuild", "esbuild");
    m.insert("swc", "SWC");
    m.insert("babel", "Babel");
    m.insert("eslint", "ESLint");
    m.insert("prettier", "Prettier");
    m.insert("tailwind", "Tailwind");
    m.insert("bootstrap", "Bootstrap");
    m.insert("graphql", "GraphQL");
    m.insert("grpc", "gRPC");
    m.insert("rest", "REST");
    m.insert("websocket", "WebSocket");
    m.insert("onnx", "ONNX");
    m.insert("llm", "LLM");
    m.insert("slm", "SLM");
    m.insert("ml", "ML");
    m.insert("ai", "AI");
    m.insert("nlp", "NLP");
    m.insert("tts", "TTS");
    m.insert("asr", "ASR");
    m.insert("stt", "STT");
    m.insert("gec", "GEC");
    m.insert("ner", "NER");
    m.insert("pos", "POS");
    m.insert("gpu", "GPU");
    m.insert("tpu", "TPU");
    m.insert("npu", "NPU");
    m.insert("fpga", "FPGA");
    m.insert("asic", "ASIC");
    m.insert("risc", "RISC");
    m.insert("cisc", "CISC");
    m.insert("arm", "ARM");
    m.insert("x86", "x86");
    m.insert("amd64", "AMD64");
    m.insert("aarch64", "AArch64");
    m.insert("riscv", "RISC-V");

    m
}

// ── tests ───────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_common_chat_abbreviations() {
        let cs = ChatSpeakNormalizer::new();
        assert_eq!(cs.proof("ur text here"), "your text here");
        assert_eq!(cs.proof("pls fix this"), "please fix this");
        assert_eq!(cs.proof("idk what u mean"), "I don't know what you mean");
        assert_eq!(cs.proof("thx for the help"), "thanks for the help");
    }

    #[test]
    fn expands_dev_abbreviations() {
        let cs = ChatSpeakNormalizer::new();
        assert_eq!(cs.proof("finished dev work"), "finished development work");
        assert_eq!(cs.proof("the impl is done"), "the implementation is done");
        assert_eq!(cs.proof("check the cfg"), "check the configuration");
    }

    #[test]
    fn capitalizes_acronyms() {
        let cs = ChatSpeakNormalizer::new();
        assert_eq!(cs.proof("gui is working"), "GUI is working");
        assert_eq!(cs.proof("tui is broken"), "TUI is broken");
        assert_eq!(cs.proof("the api and cli"), "the API and CLI");
        assert_eq!(cs.proof("use json not xml"), "use JSON not XML");
    }

    #[test]
    fn preserves_punctuation_and_spacing() {
        let cs = ChatSpeakNormalizer::new();
        let result = cs.proof("ur text here, can we do a game mode?");
        assert_eq!(result, "your text here, can we do a game mode?");
    }

    #[test]
    fn handles_mixed_case_input() {
        let cs = ChatSpeakNormalizer::new();
        assert_eq!(cs.proof("UR text"), "YOUR text");
        assert_eq!(cs.proof("Ur text"), "Your text");
    }

    #[test]
    fn handles_contractions_without_apostrophe() {
        let cs = ChatSpeakNormalizer::new();
        assert_eq!(cs.proof("dont worry"), "don't worry");
        assert_eq!(cs.proof("its working"), "it's working");
        assert_eq!(cs.proof("im here"), "I'm here");
    }

    #[test]
    fn passes_through_unknown_words() {
        let cs = ChatSpeakNormalizer::new();
        assert_eq!(cs.proof("flurbo the glaxnar"), "flurbo the glaxnar");
    }

    #[test]
    fn custom_abbreviation() {
        let cs = ChatSpeakNormalizer::new().with_abbreviation("wb", "welcome back");
        assert_eq!(cs.proof("wb to the team"), "welcome back to the team");
    }

    #[test]
    fn custom_acronym() {
        let cs = ChatSpeakNormalizer::new().with_acronym("lsp", "LSP");
        assert_eq!(cs.proof("the lsp crashed"), "the LSP crashed");
    }
}
