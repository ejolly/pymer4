def test_install():
    """
    Quick function to test installation by import a lmm object and fitting a quick model.
    """
    try:
        from pymer4.models import Lmer
        from pymer4.utils import get_resource_path
        import os
        import pandas as pd
        import warnings
        warnings.filterwarnings("ignore")
        df = pd.read_csv(os.path.join(get_resource_path(), 'sample_data.csv'))
        model = Lmer('DV ~ IV3 + (1|Group)', data=df)
        model.fit(summarize=False)
        print("Pymer4 installation working successfully!")
    except Exception as e:
        print("Error! {}".format(e))