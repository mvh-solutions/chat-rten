mod helpers;
mod rag_chat_lib;

use std::error::Error;
use std::io;
use std::io::prelude::*;

use argh;
use rten::Model;
use rten_generate::filter::Chain;
use rten_generate::sampler::Multinomial;
use rten_generate::{Generator, GeneratorUtils};
use rten_text::Tokenizer;

use helpers::{Args, encode_message};
use crate::helpers::generate_user_prompt;

fn do_prompt() -> () {
    print!("> ");
    let _ = io::stdout().flush();
}
fn main() -> Result<(), Box<dyn Error>> {
    let mut args: Args = argh::from_env();
    args.temperature = args.temperature.max(0.);

    let model = unsafe { Model::load_mmap(args.model) }?;
    let tokenizer = Tokenizer::from_file(&args.tokenizer_config)?;

    let im_end_token = tokenizer.get_token_id("<|im_end|>")?;

    let mut end_of_turn_tokens = Vec::new();
    end_of_turn_tokens.push(im_end_token);

    if let Ok(end_of_text_token) = tokenizer.get_token_id("<|endoftext|>") {
        end_of_turn_tokens.push(end_of_text_token);
    }

    // From `chat_template` in tokenizer_config.json.
    let prompt_tokens = encode_message(
        &tokenizer,
        "system\nYou are a helpful assistant. The user is translating John 3:16 in the Bible. She is translating from English, which she speaks fluently. However, she left school when she was 11 years old so her written English is limited. She likes to read short, precise answers. She likes answers that contain between one and three short paragraphs. She does not want to see the entire verse, only the parts of the verse that are relevant to the question.".to_string()
    ) ? ;

    // From Qwen2's `generation_config.json`
    let top_k = 5;

    let mut generator = Generator::from_model(&model)?
        .with_prompt(&prompt_tokens)
        .with_logits_filter(Chain::new().top_k(top_k).temperature(args.temperature))
        .with_sampler(Multinomial::new());

    let mut first_time: bool = true;
    loop {

        do_prompt();

        let mut user_input = String::new();
        let n_read = io::stdin().read_line(&mut user_input)?;
        if n_read == 0 {
            // EOF
            break;
        }
        if !first_time && user_input.clone().trim().len() == 0 {
            break;
        }
        first_time = false;

        let user_text = generate_user_prompt("JHN 3:16".to_string(), "John 3:16".to_string(), user_input);
        // println!("{}", &user_text);
        let token_ids = encode_message(
            &tokenizer,
            user_text,
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
