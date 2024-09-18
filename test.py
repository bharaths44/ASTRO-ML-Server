# import requests
# import json

# URL of the FastAPI endpoint
# url = "http://127.0.0.1:8080/predict"
url = "https://astro-ml-server.wonderfulwave-d209090d.eastus.azurecontainerapps.io/predict"
# Path to the sample CSV file
csv_file_path = "/Users/bharaths/Developer/ASTRO/Demand_forecasting/train.csv"
response_file_path = "response.json"
# Additional parameters
params = {"store_num": 1, "item_num": 1, "period_type": "M", "num_periods": 3}

# # Send the POST request with the CSV file
# with open(csv_file_path, "rb") as f:
#     files = {"file": f}
#     response = requests.post(url, files=files, params=params)

# # Check if the response is valid JSON
# try:
#     response_data = response.json()
#     decoded_json = json.loads(response_data)
# except requests.exceptions.JSONDecodeError:
#     print("Error: The response is not valid JSON.")
#     print("Response content:", response.content, response.text)
# else:
#     with open(response_file_path, "w") as f:
#         json.dump(decoded_json, f, indent=4)

#     # Print the response
#     print("Status Code:", response.status_code)
#     print("Response saved to:", response_file_path)

#     # Print the number of entries in the JSON response
#     print("Number of entries in the JSON response:", len(decoded_json))

# # Print the response
# print("Status Code:", response.status_code)
# print("Response saved to:", response_file_path)

# import json
# import pandas as pd
# import matplotlib.pyplot as plt

# # Path to the JSON file
# response_file_path = "/Users/bharaths/Developer/ASTRO/Demand_forecasting/response.json"

# # Read the JSON data from the file
# with open(response_file_path, "r") as f:
#     data = json.load(f)

# # Extract train and forecast data
# train_data = data["train"]
# forecast_data = data["forecast"]

# # Convert to pandas DataFrame
# train_df = pd.DataFrame(
#     {"date": pd.to_datetime(train_data["date"]), "sales": train_data["sales"]}
# )

# forecast_df = pd.DataFrame(
#     {"date": pd.to_datetime(forecast_data["date"]), "sales": forecast_data["sales"]}
# )

# # Plot the data
# plt.figure(figsize=(10, 6))
# plt.plot(train_df["date"], train_df["sales"], label=train_data["label"], color="blue")
# plt.plot(
#     forecast_df["date"],
#     forecast_df["sales"],
#     label=forecast_data["label"],
#     color="red",
#     linestyle="--",
# )

# # Add labels and title
# plt.xlabel("Date")
# plt.ylabel("Sales")
# plt.title("Sales Forecast")
# plt.legend()

# # Show the plot
# plt.show()import pandas as pd
import pandas as pd

# Path to the existing CSV file
csv_file_path = "/Users/bharaths/Developer/ASTRO/Demand_forecasting/train.csv"
# Path to the new CSV file for the 30% subset
subset_csv_file_path = (
    "/Users/bharaths/Developer/ASTRO/Demand_forecasting/subset_30.csv"
)

# Read the existing CSV file
df = pd.read_csv(csv_file_path)

# Ensure the 'date' column is in datetime format
df["date"] = pd.to_datetime(df["date"])

# Sort the DataFrame by date
df = df.sort_values(by="date")

# Calculate the split index for 30:70 split
split_index = int(len(df) * 0.3)

# Select the first 30% of the DataFrame
df_30 = df.iloc[:split_index]

# Save the 30% subset DataFrame to a new CSV file
df_30.to_csv(subset_csv_file_path, index=False)

print(
    f"CSV file with the first 30% of the time series data has been created at {subset_csv_file_path}"
)
