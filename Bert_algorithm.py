from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

#  Load Model and Tokenize
tokenizer = PreTrainedTokenizerFast.from_pretrained("ainize/bart-base-cnn")
model = BartForConditionalGeneration.from_pretrained("ainize/bart-base-cnn")

# Encode Input Text
input_text = '(CNN) -- South Korea launched an investigation Tuesday into reports of toxic chemicals being dumped at a former U.S. military base, the Defense Ministry said. The tests follow allegations of American soldiers burying chemicals on Korean soil. The first tests are being carried out by a joint military, government and civilian task force at the site of what was Camp Mercer, west of Seoul. "Soil and underground water will be taken in the areas where toxic chemicals were allegedly buried," said the statement from the South Korean Defense Ministry. Once testing is finished, the government will decide on how to test more than 80 other sites -- all former bases. The alarm was raised this month when a U.S. veteran alleged barrels of the toxic herbicide Agent Orange were buried at an American base in South Korea in the late 1970s. Two of his fellow soldiers corroborated his story about Camp Carroll, about 185 miles (300 kilometers) southeast of the capital, Seoul. "We\'ve been working very closely with the Korean government since we had the initial claims," said Lt. Gen. John Johnson, who is heading the Camp Carroll Task Force. "If we get evidence that there is a risk to health, we are going to fix it." A joint U.S.- South Korean investigation is being conducted at Camp Carroll to test the validity of allegations. The U.S. military sprayed Agent Orange from planes onto jungles in Vietnam to kill vegetation in an effort to expose guerrilla fighters. Exposure to the chemical has been blamed for a wide variety of ailments, including certain forms of cancer and nerve disorders. It has also been linked to birth defects, according to the Department of Veterans Affairs. Journalist Yoonjung Seo contributed to this report.'

input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate Summary Text Ids
summary_text_ids = model.generate(
    input_ids=input_ids,
    bos_token_id=model.config.bos_token_id,
    eos_token_id=model.config.eos_token_id,
    length_penalty=2.0,
    max_length=142,
    min_length=56,
    num_beams=4,
)

# Decoding Text
print(tokenizer.decode(summary_text_ids[0], skip_special_tokens=True))
