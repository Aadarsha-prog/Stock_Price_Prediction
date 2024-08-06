import asyncio
import httpx
from nepse import Client
import pandas as pd

async def fetch_data(symbol):
    async with httpx.AsyncClient() as async_client:
        client = Client(httpx_client=async_client)
        try:
            data = await client.security_client.get_company(symbol=symbol)
            return data
        except httpx.RequestError as e:
            print(f"An error occurred while requesting data for {symbol}: {e}")
            return None

async def get_stock_data(symbols):
    tasks = [fetch_data(symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks)
    return [result for result in results if result is not None]

def save_to_csv(data, filename='stock_data.csv'):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    symbols = ['NABIL', 'NLIC', 'SBI']  # Add desired stock symbols here
    loop = asyncio.get_event_loop()
    data = loop.run_until_complete(get_stock_data(symbols))
    if data:
        formatted_data = [
            {
                'symbol': d.symbol,
                'high_price': d.high_price,
                'low_price': d.low_price,
                'open_price': d.open_price,
                'close_price': d.close_price,
                'volume': d.volume,
                'date': d.date
            } for d in data
        ]
        save_to_csv(formatted_data)
    else:
        print("No data to save.")

