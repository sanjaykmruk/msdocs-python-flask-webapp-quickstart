import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
import torch

tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-7b", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-7b", device_map="auto", torch_dtype=torch.bfloat16)

PROMPT_FORMAT = """Below is an instruction that describes a task. Write a response that appropriately completes the request."""


def generate_response(instruction: str, *, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, 
                      do_sample: bool = True, max_new_tokens: int = 256, top_p: float = 0.92, top_k: int = 0, **kwargs) -> str:
    input_ids = tokenizer(PROMPT_FORMAT.format(instruction=instruction), return_tensors="pt").input_ids.to("cuda")

    # each of these is encoded to a single token
    response_key_token_id = tokenizer.encode("### Response:")[0]
    end_key_token_id = tokenizer.encode("### End")[0]

    gen_tokens = model.generate(input_ids, pad_token_id=tokenizer.pad_token_id, eos_token_id=end_key_token_id,
                                do_sample=do_sample, max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k, **kwargs)[0].cpu()

    # find where the response begins
    response_positions = np.where(gen_tokens == response_key_token_id)[0]

    if len(response_positions) >= 0:
        response_pos = response_positions[0]
        
        # find where the response ends
        end_pos = None
        end_positions = np.where(gen_tokens == end_key_token_id)[0]
        if len(end_positions) > 0:
            end_pos = end_positions[0]

        return tokenizer.decode(gen_tokens[response_pos + 1 : end_pos]).strip()

    return None


def clean_sql_string2(sql_string_input, substring="SELECT"):
    index = sql_string_input.find(substring)
    if index != -1:
        return sql_string_input[index:]
    else:
        return ""


def sql_creator(prompt_question, printing=False):
    prompt_shcema= """
    The table enterprise_azlab_prod.magratheav2 has the following schema
    |-- sales_amt: decimal(28,2) (nullable = true)
    |-- item_qty: long (nullable = true)
    |-- storeName: string (nullable = true)
    |-- department: string (nullable = true)
    |-- storeNumber: string (nullable = true)
    |-- naturalDate: date (nullable = true)
    |-- trading_area: string (nullable = true)
    |-- department_name: string (nullable = true)
    |-- business_area: string (nullable = true)
    |-- productname: string (nullable = true)
    |-- format: string (nullable = true)
    |-- division_name: string (nullable = true)
    |-- region_name: string (nullable = true)
    """

    # prompt_additional_data="""
    # The complete name of the magratheav2 table is enterprise_azlab_prod.magratheav2. Use enterprise_azlab_prod.magratheav2 as the table name.
    # """
    prompt_additional_data = """
    """

    prompt_sql = f"""
    What is a sql query for {prompt_question}
    """

    aPrompt4 = prompt_shcema + prompt_additional_data + prompt_sql
    # if printing:
    #   print(aPrompt4)

    result = generate_response(aPrompt4, model=model, tokenizer=tokenizer)
    # if printing:
    #   print(result)

    return clean_sql_string2(result), result


def qw1(prompt_question, printing=False):
    
    sql_clause, raw = sql_creator(prompt_question,True)

    sdf = spark.sql(clean_sql_string2(sql_clause))
    output_dict =sdf.toPandas().to_dict()
    # if printing:
    #   print("result in dict:" , output_dict)
    prompt_result =f"""We have the result {output_dict} from our computation."""
    aPrompt6 = prompt_result + prompt_question
    
    result2 = generate_response(aPrompt6, model=model, tokenizer=tokenizer)
    if printing: 
      print(prompt_question,  result2)
    
    return result2

def qw3(prompt_question, printing=False):
    
    sql_clause, _ = sql_creator(prompt_question,True)
    
    sql_clean = clean_sql_string2(sql_clause)
    if printing:
        print(f"SQL: {sql_clean}")
    
    sdf = spark.sql(sql_clean)
    # sdf = spark.sql(clean_sql_string2(sql_clause))
    output_dict =sdf.toPandas().to_dict()
    # if printing:
    #   print("result in dict:" , output_dict)
    prompt_result =f"""We have the result {output_dict} from our computation."""
    aPrompt6 = prompt_result + prompt_question
    aPrompt6= f"""{prompt_result} \
      {prompt_question}. \
        Provide the answer in JSON format with the following keys: 
        store_name, area_highest_sales, total_sales.
    """
    
    result2 = generate_response(aPrompt6, model=model, tokenizer=tokenizer)
    if printing: 
      print(prompt_question,  result2)
    
    return result2