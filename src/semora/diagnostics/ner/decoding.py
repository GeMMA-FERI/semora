"""Use tokenizer word IDs for aggregation instead of character-space heuristics."""


def mark_subwords(entities: list[dict], word_ids: list[int | None]) -> None:
    previous = None
    for entity in entities:
        word = word_ids[entity["index"]]
        entity["is_subword"] = word is not None and word == previous
        previous = word


def word_pipeline_class():
    # Lazy import: sample/report commands have no model dependency.
    from transformers import TokenClassificationPipeline
    from transformers.pipelines.token_classification import AggregationStrategy

    class WordPipeline(TokenClassificationPipeline):
        def gather_pre_entities(
            self,
            sentence,
            input_ids,
            scores,
            offset_mapping,
            special_tokens_mask,
            aggregation_strategy,
            word_ids=None,
            word_to_chars_map=None,
        ):
            entities = super().gather_pre_entities(
                sentence,
                input_ids,
                scores,
                offset_mapping,
                special_tokens_mask,
                AggregationStrategy.SIMPLE if word_ids is not None else aggregation_strategy,
                word_ids=word_ids,
                word_to_chars_map=word_to_chars_map,
            )
            if word_ids is not None:
                mark_subwords(entities, word_ids)
            return entities

    return WordPipeline
