use rten_text::{TokenId, Tokenizer, TokenizerError};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::error::Error;
use std::io;
use std::io::Write;
use rten_generate::{Generator, GeneratorUtils};
use rten_generate::filter::Chain;
use rten_generate::sampler::Multinomial;
use rten::Model;

pub(crate) struct ChatConfig {
    pub(crate) model_path: String,
    pub(crate) tokenizer_path: String,
    pub(crate) temperature: f32,
    pub(crate) top_k: usize
}

/// Helpers for LLM chat.

pub(crate) enum MessageChunk<'a> {
    Text(&'a str),
    Token(u32),
}

#[derive(Serialize, Deserialize)]
pub(crate) struct VerseContext {
    pub(crate) juxta: String,
    pub(crate) translations: BTreeMap<String, String>,
    pub(crate) notes: BTreeMap<String, Vec<String>>,
    pub(crate) snippets: BTreeMap<String, Vec<String>>,
}

pub(crate) fn get_end_of_turn_tokens(tokenizer: &Tokenizer) -> Vec<TokenId> {
    let im_end_token = tokenizer.get_token_id("<|im_end|>").expect("get token id for end");

    let mut end_of_turn_tokens = Vec::new();
    end_of_turn_tokens.push(im_end_token);

    if let Ok(end_of_text_token) = tokenizer.get_token_id("<|endoftext|>") {
        end_of_turn_tokens.push(end_of_text_token);
    }
    end_of_turn_tokens
}

pub(crate) fn encode_system_message(tokenizer: &Tokenizer) -> Result<Vec<u32>, TokenizerError> {
    encode_message(
        tokenizer,
        "system\nYou are a helpful assistant. The user is translating John 3:16 in the Bible. She is translating from English, which she speaks fluently. However, she left school when she was 11 years old so her written English is limited. She likes to read short, precise answers. She likes answers that contain between one and three short paragraphs. She does not want to see the entire verse, only the parts of the verse that are relevant to the question.".to_string()
    )
}
pub(crate) fn encode_message(
    tokenizer: &Tokenizer,
    user_prompt: String,
) -> Result<Vec<u32>, TokenizerError> {
    let im_start_token = tokenizer.get_token_id("<|im_start|>")?;
    let im_end_token = tokenizer.get_token_id("<|im_end|>")?;

    let mut end_of_turn_tokens = Vec::new();
    end_of_turn_tokens.push(im_end_token);

    if let Ok(end_of_text_token) = tokenizer.get_token_id("<|endoftext|>") {
        end_of_turn_tokens.push(end_of_text_token);
    }

    let mut token_ids = Vec::new();
    let chunks = &[
        MessageChunk::Token(im_start_token),
        MessageChunk::Text("user\n"),
        MessageChunk::Text(&user_prompt),
        MessageChunk::Token(im_end_token),
        MessageChunk::Text("\n"),
        MessageChunk::Token(im_start_token),
        MessageChunk::Text("assistant\n"),
    ];
    for chunk in chunks {
        match chunk {
            MessageChunk::Token(tok_id) => token_ids.push(*tok_id),
            MessageChunk::Text(text) => {
                let encoded = tokenizer.encode(*text, None)?;
                token_ids.extend(encoded.token_ids());
            }
        }
    }
    Ok(token_ids)
}

pub(crate) fn generate_user_prompt(
    _bcv: String,
    _printable_bcv: String,
    user_input: String,
) -> String {
    let verse_context_path = std::path::PathBuf::from("./test_data/JHN/ch_3/v16.json");
    let absolute_verse_context_path = std::path::absolute(&verse_context_path).expect("absolute");
    let verse_context_string =
        std::fs::read_to_string(&absolute_verse_context_path).expect("Read verse context");
    let verse_context_json: VerseContext =
        serde_json::from_str(&verse_context_string).expect("Parse verse context");
    let mut translation_contexts: Vec<String> = Vec::new();
    for (k, v) in verse_context_json.translations {
        translation_contexts.push(format!("\n- {} ({}): {}\n", "John 3:16", &k, &v));
    }
    let translation_context: String = translation_contexts.into_iter().collect();
    let juxta_context = verse_context_json.juxta.clone();

    let mut note_contexts: Vec<String> = Vec::new();
    for (k, v) in verse_context_json.notes {
        let mut numbered_notes: Vec<String> = Vec::new();
        let mut note_n = 1;
        for note in v {
            numbered_notes.push(format!("({}) {} ", note_n, &note));
            note_n += 1;
        }
        let numbered_note_string: String = numbered_notes.into_iter().collect();
        note_contexts.push(format!(
            "\n- {} from the {}: {}\n",
            "John 3:16", &k, &numbered_note_string
        ));
    }
    let note_context: String = note_contexts.into_iter().collect();

    let mut snippet_contexts: Vec<String> = Vec::new();
    for (snippet_key, snippet_value) in verse_context_json.snippets {
        let mut snippet_notes: Vec<String> = Vec::new();
        let mut note_n = 1;
        for note in snippet_value {
            snippet_notes.push(format!("({}) {} ", note_n, &note));
            note_n += 1;
        }
        let numbered_note_string: String = snippet_notes.into_iter().collect();
        snippet_contexts.push(format!(
            "\n- the word or words '{}' in {}: {}\n",
            &snippet_key, "John 3:16", &numbered_note_string
        ));
    }
    let snippet_context: String = snippet_contexts.into_iter().collect();
    format!(
        "# Source Documents\n\n{}\n\n# Greek-English Juxtalinear Translation\n\n{}\n\n# English Bible Translations\n\n{}\n{}\n# Verse Notes\n\n{}\n{}\n# Notes on key words in the verse\n\n{}\n{}\n# The user's question\n\n{}\n\n**{}**",
        "Here are some important documents. You should base your answer to her questions on these documents.",
        &juxta_context,
        "Here are different English Bible translations of the same verse. These are important. Pay attention to the names of the translations, and to the differences between the translations for this verse.",
        translation_context.as_str(),
        "Here are some notes on the whole verse. These are NOT Bible translations. The notes apply to ALL Bible translations. These notes help us to understand the Bible translations.",
        note_context.as_str(),
        "Here are some notes on important words in this verse. These notes are also NOT Bible translations. They refer to the unfoldingWord Literal Translation, but may be applied to other Bible translations.",
        snippet_context.as_str(),
        "Now answer the following question using only the documents above.",
        &user_input.trim()
    )
}

pub(crate) fn generator_from_model<'a>(model: &'a Model, tokenizer: &'a Tokenizer, top_k: usize, temperature: f32) -> Generator<'a> {
    let prompt = encode_system_message(tokenizer).expect("encode system message");
    Generator::from_model(model).expect("generator from model")
        .with_prompt(&prompt)
        .with_logits_filter(Chain::new().top_k(top_k).temperature(temperature))
        .with_sampler(Multinomial::new())
}

pub(crate) fn do_one_iteration(generator: &mut Generator, tokenizer: &Tokenizer) -> Result<bool, Box<dyn Error>> {
    let end_of_turn_tokens = get_end_of_turn_tokens(tokenizer);
    let mut user_input = String::new();
    let n_read = io::stdin().read_line(&mut user_input)?;
    if n_read == 0 {
        // EOF
        return Ok(false);
    }

    let user_text =
        generate_user_prompt("JHN 3:16".to_string(), "John 3:16".to_string(), user_input);
    // println!("{}", &user_text);
    let token_ids = encode_message(&tokenizer, user_text)?;

    generator.append_prompt(&token_ids);

    let decoder = generator
        .by_ref()
        .stop_on_tokens(&end_of_turn_tokens)
        .decode(&tokenizer);
    for token in decoder {
        let token = token?;
        print!("{}", token);
        let _ = io::stdout().flush();
    }

    println!();
    Ok(true)
}
