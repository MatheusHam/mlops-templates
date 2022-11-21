"""
  
"""


import numpy as np
import pandas as pd
from dython.nominal import associations
from pandas.api.types import is_numeric_dtype


## Standard Configs
allowed_nans_percent: int = 90,
leakage_threshold: float = 0.97,
max_rows: int = 125_000,
min_rows: int = 1_000,
correlation_threshold: float = 0.95,
percent_holdout_size: float = 0.25,
  
## Master Table params

# Setup Master Table
master_table = master_table.copy()
master_table.replace([np.inf, -np.inf], np.nan, inplace=True)
master_table.fillna(np.nan, inplace=True)

mt_cols = len(self.master_table.columns)
mt_cols_names = self.master_table.columns
mt_rows = len(self.master_table)

# Table Params
target_col_name: str = target_col_name
date_col_name: str = date_col_name
entity_col_name: str = entity_col_name
sample_weight_col_name: str = sample_weight_col_name


def sort_master_table(self):
    if self.date_col_name:
        # TimeSeries precisam estar ordenadas
        self.master_table.sort_values(
            by=[self.entity_col_name, self.date_col_name], inplace=True
        )
    else:
        self.master_table.sort_values(
            by=[self.entity_col_name], inplace=True
        )

    def spot_any_natypes(self):
        """Verifica se existe qualquer NAType nas colunas em que são proibidos:

        - Datetime Features
        - Datetime Columns
        - Sample weight Column (Se houver)
        - Target Column
        - Entity Column

        Returns
        -------
        None

        Raises
        ------
        ValueError
            {coluna} contem NATypes.
        """
        datetime_df = self.master_table.select_dtypes(
            include=[
                "datetime",
                "datetime64",
                "timedelta",
                "timedelta64",
                "datetimetz",
                "datetime64[ns]",
            ]
        )
        df_natypes = pd.DataFrame({"values": datetime_df.isna().sum()})

        if self.sample_weight_col_name:
            if self.master_table[self.sample_weight_col_name].isna().sum() > 0:
                raise ValueError(
                    "Sua coluna de sample weight:"
                    f" {self.sample_weight_col_name} contem NAType. Por favor,"
                    " remova as amostrar ou preencha e tente novamente."
                )
        for col in [self.target_col_name, self.entity_col_name]:
            if col not in self.master_table.columns:
                raise ValueError(
                    f"A sua tabela não possui a coluna '{col}', revise sua"
                    " master table."
                )

        if self.master_table[self.entity_col_name].isna().sum() > 0:
            raise ValueError(
                f"A sua coluna entity: {self.entity_col_name} contem NATypes."
                "Por favor, remova as amostrar ou preencha e tente novamente."
            )

        if self.master_table[self.target_col_name].isna().sum() > 0:
            raise ValueError(
                f"Sua coluna de target: {self.target_col_name} contem NATypes."
                "Por favor, remova as amostrar ou preencha e tente novamente."
            )

        if np.any(df_natypes > 0):
            na_cols = df_natypes.index[df_natypes["values"] > 0].tolist()

            raise ValueError(
                f"Sua coluna de datetime: {na_cols} contem NATypes."
                "Por favor, remova as amostrar ou preencha e tente novamente."
            )

        self._logger.info("NATypes conferidos.")


    def reduce_mem_usage(self, df: pd.DataFrame, reduce_df_mem: bool):
        """Itera por todas as colunas do dataframe, atua na redução de memória
        alocada para os tipos numéricos. e.g Substitui int64 por int32 se o
        conteúdo não exigir um espaço maior do que int32.

        Parameters
        ----------
        df : pd.DataFrame
            Master table
        reduce_df_mem : bool
            True or False
        """
        if reduce_df_mem:
            start_mem = df.memory_usage().sum() / 1024 ** 2
            self._logger.info(
                f"Memória utilizada pela master table {start_mem:.2f} MB"
            )

            numerical_coluns = set(df.select_dtypes(np.number).columns) - set(
                [
                    self.entity_col_name,
                    self.target_col_name,
                    self.date_col_name,
                ]
            )

            for col in numerical_coluns:
                col_type = df[col].dtype.name

                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == "int":
                    if (
                        c_min > np.iinfo(np.int8).min
                        and c_max < np.iinfo(np.int8).max
                    ):
                        df[col] = df[col].astype(np.int32)
                    elif (
                        c_min > np.iinfo(np.int16).min
                        and c_max < np.iinfo(np.int16).max
                    ):
                        df[col] = df[col].astype(np.int32)
                    elif (
                        c_min > np.iinfo(np.int32).min
                        and c_max < np.iinfo(np.int32).max
                    ):
                        df[col] = df[col].astype(np.int32)
                    elif (
                        c_min > np.iinfo(np.int64).min
                        and c_max < np.iinfo(np.int64).max
                    ):
                        df[col] = df[col].astype(np.int64)
                else:
                    if (
                        c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max
                    ):
                        df[col] = df[col].astype(np.float32)
                    elif (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                    ):
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)

            end_mem = df.memory_usage().sum() / 1024 ** 2
            self._logger.info(
                f"Memória utilizada após otimização: {end_mem:.2f} MB"
            )
            self._logger.info(
                "Reduzida em {:.1f}%".format(
                    100 * (start_mem - end_mem) / start_mem
                )
            )
            self.master_table = df
        else:
            self._logger.warning(
                "You are using reduce_df_mem: False, try to use with True for"
                " better performance"
            )

    def spot_natypes_threshold(self):
        f"""Raises error in case any column of our master_table
        has nans above {self.allowed_nans_percent}%."""
        self._logger.info("Checking for natypes")
        df_natypes = pd.DataFrame({"values": self.master_table.isna().mean()})
        if np.any((100 * df_natypes) > self.allowed_nans_percent):
            raise ValueError(
                f"""Cols were identified to have NATypes above the limit:
                {self.allowed_nans_percent}% \n
                cols: {df_natypes.index[df_natypes["values"] > (self.allowed_nans_percent / 100)].tolist()}"""
            )

    def spot_leakage(self) -> dict:
        """Verifica possiveis vazamentos de dados entre suas features e o
        target. Previne situações em que o usuário deixa rastros de features
        que possivelmente constituem a construção da váriavel resposta.

        !!!tip
        Exemplo:
        Imagine um conjunto de features para definir uma faixa de valor para um imovel, onde
        escolhemos labels barato, caro e carissimo. Juntamos todo o conjunto de features para treinar o
        modelo e por acidente colocamos o IPTU como uma feature. Ao treinar o modelo teriamos um
        resultado excelente, porem totalmente enviesado, pois sabemos que o valor cobrado pelo IPTU
        está diretamente atrelado ao valor do imóvel.
        """
        self._logger.info(
            f"Procurando por vazamentos, isso pode levar um tempo..."
        )

        features = list(
            set(self.master_table.columns)
            - set(
                [
                    self.entity_col_name,
                    self.date_col_name,
                ]
            )
        )
        
        
corrs = associations(
    self.master_table[features][: (round(len(self.master_table) / 5))],
    clustering=True,
    display_rows=[self.target_col_name],
    compute_only=True,
)


corrs_dict = corrs["corr"].to_dict(orient="list")
leaky_features = {}
for key, value in corrs_dict.items():
    if key != self.target_col_name:
        if (value[0] > self.leakage_threshold) or (
            value[0] < -self.leakage_threshold
        ):
            leaky_features[key] = value
            
if leaky_features:
    raise ValueError(
        "Sua master table apresentou vazamento de dados do target nas"
        f" seguintes colunas: {leaky_features.keys()}"
    )

