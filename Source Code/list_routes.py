from app import app
print('ENDPOINTS:', sorted([rule.endpoint for rule in app.url_map.iter_rules()]))
print('RULES:')
for rule in app.url_map.iter_rules():
    print('  ', rule)
