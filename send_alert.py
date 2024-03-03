import requests

url = 'https://discord.com/api/v9/channels/1213789454603522098/messages'
payload = {
    "content" : "Test Message From Python."
}

headers = {
    "Authorization" : "OTUxODg0MTM5MTgxNzY4NzE0.G4VcoH.5Ke8EuvH0zMnEESxKHiegNau6KYQYbX5AZx0jI"
}
res = requests.post(url, payload, headers = headers)
