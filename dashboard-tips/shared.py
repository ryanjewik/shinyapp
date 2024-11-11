from pathlib import Path

import pandas as pd

app_dir = Path(__file__).parent
tips = pd.read_csv(app_dir / "tips.csv")
new_df = pd.read_csv(app_dir / "new_df.csv")