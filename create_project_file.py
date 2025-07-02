import pandas as pd

# Create project details based on sample data
project_data = []

# Add projects to match your models (1 to 103)
for i in range(1, 104):
    project_data.append({
        'Project ID': i,
        'Project Brand': 'ATMOZ' if i % 2 == 0 else 'MODIZ',
        'Project Type': 'LOW RISE' if i % 2 == 0 else 'HIGH RISE',
        'Starting Price': 2000000 + (i * 50000),
        'Location': 'ลาดพร้าว' if i % 2 == 0 else 'รัชดา'
    })

df = pd.DataFrame(project_data)
df.to_excel('notebooks/project_info/ProjectID_Detail.xlsx', index=False, engine='openpyxl')
print(f'Created ProjectID_Detail.xlsx with {len(df)} projects')
