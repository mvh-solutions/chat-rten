mod helpers;
use std::collections::BTreeMap;
use std::error::Error;
use std::io;
use std::io::prelude::*;

use argh;
use rten::Model;
use rten_generate::filter::Chain;
use rten_generate::sampler::Multinomial;
use rten_generate::{Generator, GeneratorUtils};
use rten_text::Tokenizer;
use serde_json;

use helpers::{Args, MessageChunk, encode_message};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct VerseContext {
    juxta: String,
    translations: BTreeMap<String, String>,
    notes: BTreeMap<String, Vec<String>>,
    snippets: BTreeMap<String, Vec<String>>
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args: Args = argh::from_env();
    args.temperature = args.temperature.max(0.);

    let model = unsafe { Model::load_mmap(args.model) }?;
    let tokenizer = Tokenizer::from_file(&args.tokenizer_config)?;

    let im_start_token = tokenizer.get_token_id("<|im_start|>")?;
    let im_end_token = tokenizer.get_token_id("<|im_end|>")?;

    let mut end_of_turn_tokens = Vec::new();
    end_of_turn_tokens.push(im_end_token);

    if let Ok(end_of_text_token) = tokenizer.get_token_id("<|endoftext|>") {
        end_of_turn_tokens.push(end_of_text_token);
    }
    let verse_context_path = std::path::PathBuf::from("./test_data/JHN/ch_3/v16.json");
    let absolute_verse_context_path = std::path::absolute(&verse_context_path).expect("absolute");
    let verse_context_string = std::fs::read_to_string(&absolute_verse_context_path).expect("Read verse context");
    let verse_context_json: VerseContext = serde_json::from_str(&verse_context_string).expect("Parse verse context");
    let mut translation_contexts: Vec<String> = Vec::new();
    for (k, v) in verse_context_json.translations {
        translation_contexts.push(format!("\nHere is {} from the {}: {}\n", "John 3:16", &k, &v ));
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
        note_contexts.push(format!("\nHere are notes about {} from the {}: {}\n", "John 3:16", &k, &numbered_note_string));
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
        snippet_contexts.push(format!("\nHere are notes about the word or words '{}' in {}: {}\n", &snippet_key, "John 3:16", &numbered_note_string));
    }
    let snippet_context: String = snippet_contexts.into_iter().collect();

    // From `chat_template` in tokenizer_config.json.
    let prompt_tokens = encode_message(
        &tokenizer,
        &[
            MessageChunk::Token(im_start_token),
            MessageChunk::Text(
                "system\nYou are a helpful assistant. The user, Jenny, is translating John 3:16 in the Bible. Jenny is translating from English, which she speaks fluently. However, she left school when she was 11 so her written English is limited. She likes to read short, precise answers. She does not want to see the entire verse, only the parts of the verse that are relevant to the question. Your answer should consist of between one and three short paragraphs.",
            ),
            MessageChunk::Token(im_end_token),
        ],
    )?;

    // From Qwen2's `generation_config.json`
    let top_k = 5;

    let mut generator = Generator::from_model(&model)?
        .with_prompt(&prompt_tokens)
        .with_logits_filter(Chain::new().top_k(top_k).temperature(args.temperature))
        .with_sampler(Multinomial::new());

    loop {
        print!("> ");
        let _ = io::stdout().flush();

        let mut user_input = String::new();
        let n_read = io::stdin().read_line(&mut user_input)?;
        if n_read == 0 {
            // EOF
            break;
        }

        // From `chat_template` in tokenizer_config.json.
        let token_ids = encode_message(
            &tokenizer,
            &[
                MessageChunk::Token(im_start_token),
                MessageChunk::Text("user\n"),
                MessageChunk::Text("Here are some documents that Jenny thinks are important. You should base your answer to her questions on this context.\n\n"),
		MessageChunk::Text(&juxta_context),
                MessageChunk::Text("Here are different English Bible translations of the same verse. These are important. Pay attention to the names of the translations, and to the differences between the translations for this verse.\n"),
                MessageChunk::Text(translation_context.as_str()),
                MessageChunk::Text("\nHere are some notes on the whole verse. These are NOT Bible translations. The notes apply to ALL Bible translations. These notes help us to understand the Bible translations.\n"),
                MessageChunk::Text(note_context.as_str()),
                MessageChunk::Text("\nHere are some notes on important words in this verse. These notes are also NOT Bible translations. They refer to the unfoldingWord Literal Translation, but may be applied to other Bible translations.\n\n"),
                MessageChunk::Text(snippet_context.as_str()),
                MessageChunk::Text("Now answer the following question using the context documents above. "),
                MessageChunk::Text(&user_input),
                MessageChunk::Token(im_end_token),
                MessageChunk::Text("\n"),
                MessageChunk::Token(im_start_token),
                MessageChunk::Text("assistant\n"),
            ],
        )?;

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
    }

    Ok(())
}
