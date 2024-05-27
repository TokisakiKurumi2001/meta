# Transformers v4.36
from transformers import LlamaForCausalLM
from meta_head import MetaheadConfig, MetaheadModel

if __name__ == "__main__":
    teacher = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')
    embed_dict = teacher.get_input_embeddings()
    vocab_head = teacher.get_output_embeddings()

    meta_config = MetaheadConfig.from_pretrained('meta-llama/Llama-2-7b-hf')
    meta_model = MetaheadModel(meta_config)
    meta_model.set_input_embeddings(embed_dict)
    meta_model.set_output_embeddings(vocab_head)
    meta_model.save_pretrained('meta-head')