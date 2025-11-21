# Real-Time Dashboard Walkthrough

I have implemented real-time capabilities by connecting the **Chatbot** and the **Dashboard**.

## How it Works
1.  **Lead Capture**: The chatbot (`src/Chatbot.py`) saves every completed lead interaction to `data/raw/new_leads.csv`.
2.  **Data Display**: The dashboard (`src/Dashboard.py`) reads this CSV file and displays the new leads in a dedicated "Leads en Tiempo Real" section.
3.  **Updates**: You can click the **"🔄 Actualizar Datos en Tiempo Real"** button on the dashboard to see new leads immediately without restarting the server.

## Verification Steps

1.  **Start the Dashboard**:
    ```bash
    streamlit run src/Dashboard.py
    ```

2.  **Start the Chatbot** (in a new terminal):
    ```bash
    streamlit run src/Chatbot.py
    ```

3.  **Simulate a Lead**:
    - Interact with the chatbot.
    - Provide your name, industry, etc.
    - Finish the conversation.

4.  **Check the Dashboard**:
    - Go to the dashboard tab.
    - Click the **"🔄 Actualizar Datos en Tiempo Real"** button.
    - You should see your new lead appear in the table!

## Changes Made
- **`src/Chatbot.py`**: Added `save_lead` function to append data to `data/raw/new_leads.csv`.
- **`src/Dashboard.py`**: Added logic to read `data/raw/new_leads.csv` and a refresh button.
