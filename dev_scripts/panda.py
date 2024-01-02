import pandas as pd

# Create a sample DataFrame with a DatetimeIndex
df = pd.DataFrame(
    index=pd.date_range(start="2022-04-01", end="2023-03-31", freq="M"),
    data={"value": range(12)},
)

# Display the original DataFrame
print("Original DataFrame:")
print(df)

# Change the frequency of the DatetimeIndex
df_resampled = df.asfreq(freq="AS")

# Display the DataFrame with the new frequency
print("\nDataFrame with new Frequency:")
print(df_resampled)
