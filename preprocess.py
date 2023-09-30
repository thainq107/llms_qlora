import os
import json

from datasets import load_dataset

id2label = {
    '0': 'negative',
    '1': 'positive'
}

def load_data_from_datahub(
        dataset_name, 
        save_data_dir,
        prompt=""
    ):
    raw_dataset = load_dataset(dataset_name)

    for data_type in raw_dataset:
        examples = raw_dataset[data_type]
        sentences = []
        for _, sentence in enumerate(examples["text"]):
            sentence = prompt + sentence + ". Answer:"
            sentences.append(sentence)
        
        labels = examples["feeling"]
        labels = [id2label[str(label)] for label in labels]

        save_data_file = os.path.join(save_data_dir, f"{data_type}.jsonl")
        print(f"Write into ... {save_data_file}")
        with open(save_data_file, "w") as f:
            for sentence, label in zip(sentences, labels):
                data = {
                    "sentence": sentence,
                    "label": label
                }
                print(json.dumps(data, ensure_ascii=False), file=f)