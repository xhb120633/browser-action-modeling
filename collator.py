"""Process screenshots and supervise every assistant action in a conversation."""

import torch


class ActionCollator:
    def __init__(self, processor):
        self.processor = processor
        processor.tokenizer.padding_side = "right"
        self.assistant_header = processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        self.turn_end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def __call__(self, examples):
        texts = [
            self.processor.apply_chat_template(example["messages"], tokenize=False)
            for example in examples
        ]
        images = [image for example in examples for image in example["images"]]
        batch = self.processor(text=texts, images=images, padding=True, return_tensors="pt")

        # Find action spans after image placeholders expand into visual tokens.
        # Include the end-of-turn token so training teaches the model where to stop.
        labels = torch.full_like(batch["input_ids"], -100)
        header_length = len(self.assistant_header)
        for row, input_ids in enumerate(batch["input_ids"].tolist()):
            position = 0
            while position < len(input_ids) - header_length:
                if input_ids[position:position + header_length] == self.assistant_header:
                    start = position + header_length
                    end = input_ids.index(self.turn_end, start) + 1
                    labels[row, start:end] = batch["input_ids"][row, start:end]
                    position = end
                else:
                    position += 1
        batch["labels"] = labels
        return batch
