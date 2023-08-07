import pandas as pd

def update_dataframe(prr, investment_objective, investment_strategy, investment_policy, ESG, fund_name):

    # Define the columns
    columns = ['prr', 'fund_name', 'investment_objective', 'investment_strategy', 'investment_policy', 'ESG']
    
    # Create an empty DataFrame with the specified columns
    df = pd.DataFrame(columns=columns)
    
    # Create a dictionary for the new record
    new_record = {
        'prr': prr,
        'fund_name': fund_name,
        'investment_objective': investment_objective,
        'investment_strategy': investment_strategy,
        'investment_policy': investment_policy,
        'ESG': ESG
    }

    # Check if a record with the given prr already exists
    if prr in df['prr'].values:
        # Update the existing record
        idx = df[df['prr'] == prr].index[0]
        df.loc[idx] = new_record
    else:
        # Convert the new record to a DataFrame
        new_df = pd.DataFrame([new_record])
        # Concatenate the existing DataFrame with the new record
        df = pd.concat([df, new_df], ignore_index=True)

    return df