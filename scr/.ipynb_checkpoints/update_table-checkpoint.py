import pandas as pd

# Define the columns
def update_dataframe(df, Product_reference_num, investment_objective, investment_strategy, investment_policy, ESG, fund_name):
    
    # Create a dictionary for the new record
    new_record = {
        'Product_reference_num': Product_reference_num,
        'fund_name': fund_name,
        'investment_objective': investment_objective,
        'investment_strategy': investment_strategy,
        'investment_policy': investment_policy,
        'ESG': ESG
    }

    # Check if a record with the given prr already exists
    if Product_reference_num in df['Product_reference_num'].values:
        # Update the existing record
        idx = df[df['Product_reference_num'] == Product_reference_num].index[0]
        df.loc[idx] = new_record
    else:
        # Convert the new record to a DataFrame
        new_df = pd.DataFrame([new_record])
        # Concatenate the existing DataFrame with the new record
        df = pd.concat([df, new_df], ignore_index=True)

    return df