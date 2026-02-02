use argh::FromArgs;
use rten_text::{Tokenizer, TokenizerError};

/// Helpers for LLM chat.
#[derive(FromArgs)]
pub(crate) struct Args {
    /// input model
    #[argh(positional)]
    pub(crate) model: String,

    /// tokenizer.json file
    #[argh(positional)]
    pub(crate) tokenizer_config: String,

    /// generation temperature (must be >= 0, default: 0.7). Smaller values make output less "creative" by concentrating the probability distribution more. A value of 0.0 causes sampling to be greedy.
    #[argh(option, short = 't', default = "0.5")]
    pub(crate) temperature: f32,
}

pub(crate) enum MessageChunk<'a> {
    Text(&'a str),
    Token(u32),
}

pub(crate) fn encode_message(
    tokenizer: &Tokenizer,
    chunks: &[crate::MessageChunk],
) -> Result<Vec<u32>, TokenizerError> {
    let mut token_ids = Vec::new();
    for chunk in chunks {
        match chunk {
            crate::MessageChunk::Token(tok_id) => token_ids.push(*tok_id),
            crate::MessageChunk::Text(text) => {
                let encoded = tokenizer.encode(*text, None)?;
                token_ids.extend(encoded.token_ids());
            }
        }
    }
    Ok(token_ids)
}
