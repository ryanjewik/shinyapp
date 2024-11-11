from pathlib import Path

import pandas as pd

app_dir = Path(__file__).parent
new_df = pd.read_csv(app_dir / "new_df.csv")