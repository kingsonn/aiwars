import requests

print("Checking your current public IP addresses...\n")

try:
    # Check IPv4
    response = requests.get('https://api.ipify.org?format=json', timeout=5)
    ipv4 = response.json()['ip']
    print(f"Your current IPv4: {ipv4}")
except Exception as e:
    print(f"Could not get IPv4: {e}")

try:
    # Check IPv6
    response = requests.get('https://api64.ipify.org?format=json', timeout=5)
    ipv6 = response.json()['ip']
    print(f"Your current IPv6: {ipv6}")
except Exception as e:
    print(f"Could not get IPv6: {e}")

print("\n⚠️  Compare these with the IPs you sent to Weex")
print("If they're different, send the new IPs to Weex for whitelisting")
