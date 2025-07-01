import openai
import json
import backoff
import numpy as np
import re
import os
import argparse

def extract_options_from_metas(metas):
    pattern = re.compile(r'^\s*\d+\.?\s*')
    options_list = []
    answer_indices = []
    domains = []
    
    for meta in metas:
        choices_str = meta[6]
        domains.append(meta[3])
        raw_options = choices_str.split('\n')
        cleaned_options = [pattern.sub("", option).strip().lower() 
                           for option in raw_options if option.strip()]
        options_list.append(cleaned_options)
        
        try:
            ans_idx = int(meta[5]) - 1
        except Exception:
            ans_idx = -1
        answer_indices.append(ans_idx)
    
    return options_list, answer_indices, domains

def generate_prompt(options, domain):
    prompt_lines = [
        "You will output a JSON object. Each key is an action name below,",
        "and its value is a list of AT LEAST 10 atomic, visually distinctive subaction phrases.",
        "Each phrase should capture one clear phase of the motion and reference",
        "objects, environment, or movement cues in the " + domain + " context.",
        "Phrases should be concise (a few words each) and self-contained such that they could match frames of motion independently.",
        "",
        "Actions to decompose:"
    ]
    for opt in options:
        prompt_lines.append(f"- {opt} in {domain}")
    prompt_lines += [
        "",
        "Output ONLY the JSON object mapping each action to its subactions."
    ]
    return "\n".join(prompt_lines)

@backoff.on_exception(
    backoff.expo,
    openai.RateLimitError,
    max_tries=6
)
def generate_sequence(client, prompt, temp=1.0):
    response = client.chat.completions.create(
        model="chatgpt-4o-latest",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
        temperature=temp
    )
    content = response.to_dict()['choices'][0]['message']['content']
    json_match = re.search(r'({.*})', content, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        return json.loads(json_str)
    return content

def generate_subactions(client, dataset, cleaned_options, answers, domains, temp=1.0, output_json="generated_subactions.json"):
    results = []
    processed_ids = set()

    if os.path.exists(output_json):
        try:
            with open(output_json, "r") as f:
                results = json.load(f)
                processed_ids = {item["id"] for item in results if "id" in item}
            print(f"Resuming from {len(processed_ids)} already processed samples.")
        except Exception as e:
            print("Error loading previous results; starting from scratch:", e)

    num_samples = len(dataset)
    for i, (item, opts, answ, dm) in enumerate(zip(dataset, cleaned_options, answers, domains)):
        sample_id = item[1]
        if sample_id in processed_ids:
            continue
        try:
            youtube_id = item[0]
            action = item[2]
            domain = item[3]
            question = item[4]

            print(f"Processing sample {i+1}/{num_samples} (ID: {sample_id}, Action: '{action}')")
            prompt = generate_prompt(options=opts, domain=dm)
            generated_json = generate_sequence(client=client, prompt=prompt, temp=temp)

            result = {
                "youtube_id": youtube_id,
                "id": sample_id,
                "domain": domain,
                "action": action,
                "question": question,
                "cleaned_options": opts,
                "correct_answer_index": answ,
                "subactions": generated_json
            }
            results.append(result)
            processed_ids.add(sample_id)

            if (i + 1) % 10 == 0:
                with open(output_json, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"Checkpoint: saved {len(results)} samples.")
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Subactions generated and saved to {output_json}.")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate subaction descriptions using GPT.")
    parser.add_argument("--input", required=True, help="Path to processed dataset (.npy)")
    parser.add_argument("--output", required=True, help="Path to save the generated subactions (.json)")
    parser.add_argument("--apikey", required=True, help="OpenAI API key")
    parser.add_argument("--temp", type=float, default=1.0, help="Sampling temperature for generation (default=1.0)")
    args = parser.parse_args()

    # Load dataset
    dataset = np.load(args.input, allow_pickle=True)
    dataset = [row[:9] for row in dataset if row is not None]
    dataset = np.array(dataset, dtype=object)

    cleaned_options, answer_idx, domains = extract_options_from_metas(dataset)

    client = openai.OpenAI(api_key=args.apikey)

    generate_subactions(
        client=client,
        dataset=dataset,
        cleaned_options=cleaned_options,
        answers=answer_idx,
        domains=domains,
        temp=args.temp,
        output_json=args.output
    )
