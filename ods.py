import os,sys
from pathlib import Path
import shutil
import tempfile
import zipfile
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple
sys.path.append(str(Path(__file__).parent.parent.parent))
# Third-party imports
import pandas as pd
from dateutil.relativedelta import relativedelta
from google.cloud import storage
# Local imports
from lib.BqClient import BqClient
from lib.BqStorageClient import BqStorageClient
from lib.CustomLogger import CustomLogger
from lib.utils import ecrire_param, lire_param
from src.Macro.util_dec_aco_recup_id_trt import UTIL_DEC_ACO_RECUP_ID_TRT
from src.Macro.util_dec_aco_alim_suivi_trt import util_dec_aco_alim_suivi_trt
from src.Macro.util_dec_aco_alim_suivi_trt_fin import UTIL_DEC_ACO_ALIM_SUIVI_TRT_FIN
from src.Macro.util_dec_aco_mail_envoi import util_dec_aco_mail_envoi

# Configuration file paths
ENV_FILE = "env.json"
INPUT_FILE = "input.json"
CONTEXT_FILE = "context.json"

def process_table_validation(
    df: pd.DataFrame,
    field_info: Dict[str, Dict[str, Any]],
    nb_var_c: int,
    nb_var_d: int,
    nb_var_n: int,
    nb_cle_c: int,
    nb_cle_n: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process data validation and type conversion for a staging table.
    
    This function performs type conversions and validation on staging data.
    Fields marked with FLG_EXCLUSION_DIFF='O' are excluded from error calculations
    as they should not trigger validation errors or be considered in change detection
    during historization.
    
    Args:
        df: DataFrame containing staging data
        field_info: Dictionary with field metadata (includes flg_exclusion flag)
        nb_var_c: Number of character variables
        nb_var_d: Number of date variables
        nb_var_n: Number of numeric variables
        nb_cle_c: Number of character key variables
        nb_cle_n: Number of numeric key variables
    
    Returns:
        Tuple of (df_ods, df_erreur, df_err_char, df_err_date, df_err_num, df_err_cle)
            - df_ods: Valid records after validation
            - df_erreur: All error records
            - df_err_char: Character validation errors
            - df_err_date: Date validation errors
            - df_err_num: Numeric validation errors
            - df_err_cle: Key validation errors
    """
    # Create copy for processing
    df_work: pd.DataFrame = df.copy()
    
    # Initialize error tracking columns
    for field_name in field_info.keys():
        df_work[f'ERR_{field_name}'] = 0
    
    # Track fields excluded from diff operations
    excluded_fields: List[str] = [name for name, info in field_info.items() if info.get('flg_exclusion') == 'O']
    
    # Process each field
    for field_name, info in field_info.items():
        field_type: str = info['type']
        symbol_mt_negatif: Optional[str] = info['symbol_mt_negatif']
        fmt_decimal: Optional[int] = info['fmt_decimal']
        
        # Handle negative amount symbols
        if symbol_mt_negatif and pd.notna(symbol_mt_negatif) and symbol_mt_negatif.strip():
            df_work[field_name] = df_work[field_name].astype(str)
            mask = df_work[field_name].str.startswith(symbol_mt_negatif)
            df_work.loc[mask, field_name] = '-' + df_work.loc[mask, field_name].str[len(symbol_mt_negatif):]
            df_work.loc[~mask, field_name] = '+' + df_work.loc[~mask, field_name]
            
            # Remove if only sign
            df_work.loc[df_work[field_name].str.strip().isin(['+', '-']), field_name] = ''
            df_work[field_name] = df_work[field_name].str.replace(r'[^\d\-\+\.]', '', regex=True)
        
        # Handle date fields with '00000000'
        if field_type == 'D':
            df_work.loc[df_work[field_name] == '00000000', field_name] = ''
        
        # Convert to appropriate type
        original_col = df_work[field_name].copy()
        
        if field_type == 'N':
            # Numeric conversion
            df_work[field_name] = pd.to_numeric(df_work[field_name], errors='coerce')
            
            # Apply decimal formatting
            if fmt_decimal and fmt_decimal > 0:
                df_work[field_name] = df_work[field_name] / fmt_decimal
            
            # Mark errors
            mask_error = (df_work[field_name].isna()) & (original_col.astype(str).str.strip() != '')
            df_work.loc[mask_error, f'ERR_{field_name}'] = 1
            
        elif field_type == 'D':
            # Date conversion
            df_work[field_name] = pd.to_datetime(df_work[field_name], errors='coerce', format='%Y%m%d')
            
            # Mark errors
            mask_error = (df_work[field_name].isna()) & (original_col.astype(str).str.strip() != '')
            df_work.loc[mask_error, f'ERR_{field_name}'] = 1
            
        elif field_type == 'C':
            # Character - check if conversion changed value
            df_work[field_name] = df_work[field_name].astype(str)
            mask_error = df_work[field_name] != original_col.astype(str)
            df_work.loc[mask_error, f'ERR_{field_name}'] = 1
    
    # Calculate error totals (excluding fields marked for exclusion from diff)
    if nb_var_d > 0:
        date_err_cols: List[str] = [
            f'ERR_{name}' for name, info in field_info.items()
            if info['type'] == 'D' and info.get('flg_exclusion') != 'O'
        ]
        df_work['ALL_ERR_DATE'] = df_work[date_err_cols].sum(axis=1) if date_err_cols else 0
    else:
        df_work['ALL_ERR_DATE'] = 0
    
    if nb_var_n > 0:
        num_err_cols: List[str] = [
            f'ERR_{name}' for name, info in field_info.items()
            if info['type'] == 'N' and info.get('flg_exclusion') != 'O'
        ]
        df_work['ALL_ERR_NUM'] = df_work[num_err_cols].sum(axis=1) if num_err_cols else 0
    else:
        df_work['ALL_ERR_NUM'] = 0
    
    if nb_var_c > 0:
        char_err_cols: List[str] = [
            f'ERR_{name}' for name, info in field_info.items()
            if info['type'] == 'C' and info.get('flg_exclusion') != 'O'
        ]
        df_work['ALL_ERR_CHAR'] = df_work[char_err_cols].sum(axis=1) if char_err_cols else 0
    else:
        df_work['ALL_ERR_CHAR'] = 0
    
    # Key errors
    if nb_cle_c > 0 or nb_cle_n > 0:
        key_err_cols: List[str] = [f'ERR_{name}' for name, info in field_info.items() if info['flg_cle'] == 'O']
        df_work['ALL_CLE_C_ERR'] = 0
        df_work['ALL_CLE_N_ERR'] = 0
        
        for name, info in field_info.items():
            if info['flg_cle'] == 'O':
                if info['type'] == 'C':
                    df_work['ALL_CLE_C_ERR'] += df_work[f'ERR_{name}']
                elif info['type'] == 'N':
                    df_work['ALL_CLE_N_ERR'] += df_work[f'ERR_{name}']
    else:
        df_work['ALL_CLE_C_ERR'] = 0
        df_work['ALL_CLE_N_ERR'] = 0
    
    # Separate error records
    df_err_char: pd.DataFrame = df_work[df_work['ALL_ERR_CHAR'] > 0].copy()
    df_err_date: pd.DataFrame = df_work[df_work['ALL_ERR_DATE'] > 0].copy()
    df_err_num: pd.DataFrame = df_work[df_work['ALL_ERR_NUM'] > 0].copy()
    df_err_cle: pd.DataFrame = df_work[(df_work['ALL_CLE_N_ERR'] > 0) | (df_work['ALL_CLE_C_ERR'] > 0)].copy()
    
    # All errors
    df_erreur: pd.DataFrame = df_work[
        (df_work['ALL_ERR_CHAR'] > 0) |
        (df_work['ALL_ERR_DATE'] > 0) |
        (df_work['ALL_ERR_NUM'] > 0) |
        (df_work['ALL_CLE_C_ERR'] > 0) |
        (df_work['ALL_CLE_N_ERR'] > 0)
    ].copy()
    
    # Valid records
    df_ods: pd.DataFrame = df_work[
        (df_work['ALL_ERR_CHAR'] == 0) &
        (df_work['ALL_ERR_DATE'] == 0) &
        (df_work['ALL_ERR_NUM'] == 0) &
        (df_work['ALL_CLE_C_ERR'] == 0) &
        (df_work['ALL_CLE_N_ERR'] == 0)
    ].copy()
    
    # Keep only original columns in df_ods
    original_cols: List[str] = list(field_info.keys())
    df_ods = df_ods[original_cols]
    
    return df_ods, df_erreur, df_err_char, df_err_date, df_err_num, df_err_cle


def util_dec_aco_charg_ods() -> None:
    """
    Main function to load data from staging tables (TAMPON) to ODS tables.
    Performs data validation, duplicate detection, and creates versioned ODS tables.
    
    This function:
    - Loads configuration from JSON files
    - Retrieves treatment IDs and processing dates
    - Processes each staging table for validation and type conversion
    - Creates or updates ODS tables with versioning
    - Handles confidentiality levels
    - Manages file archiving and error tracking
    
    Returns:
        None
    """
    # Load environment configuration
    try:
        env: Dict[str, Any] = lire_param(ENV_FILE)
    except FileNotFoundError as e:
        print(f"Error loading environment configuration: {str(e)}")
        raise
    
    # Extract environment variables
    project_id: str = env["project_id"]
    logger_name: str = env["logger_name"]
    team_name: str = env["team_name"]
    program_name: str = "util_dec_aco_charg_ods"
    
    # Initialize logger
    logger = CustomLogger(
        project_id=project_id,
        logger_name=logger_name,
        program_name=program_name,
        team_name=team_name,
    )
    
    logger.info("Starting CHARGEMENT_ODS process")
    
    # Initialize BigQuery client
    client = BqClient(
        project_id=project_id,
        logger_name=logger_name,
        program_name=program_name,
        team_name=team_name,
    )
    
    # Load context parameters
    try:
        params: Dict[str, Any] = lire_param(CONTEXT_FILE)
    except FileNotFoundError as e:
        logger.error(f"Error loading context parameters: {str(e)}")
        raise
    
    # # Call global variables initialization (imported macro)
    # try:
    #     util_variables_globales()
    # except Exception as e:
    #     logger.error(f"Error initializing global variables: {str(e)}")
    #     raise
    
    # Retrieve treatment ID (imported macro)
    try:
        UTIL_DEC_ACO_RECUP_ID_TRT()
    except Exception as e:
        logger.error(f"Error retrieving treatment ID: {str(e)}")
        raise
    
    # Reload params after macro calls that may have updated them
    try:
        params = lire_param(CONTEXT_FILE)
    except FileNotFoundError as e:
        logger.error(f"Error reloading context parameters: {str(e)}")
        raise
    
    MV_ID_TRT: Optional[int] = params.get("MV_ID_TRT")
    MV_DAT_BATCH_TRT: Optional[int] = params.get("MV_DAT_BATCH_TRT")
    MV_DAT_BATCH_TRT_DATE = params.get("MV_DAT_BATCH_TRT_DATE")
    lib_dec_aco_suivi: Optional[str] = params.get("lib_dec_aco_suivi")
    lib_dec_aco_prm: Optional[str] = params.get("lib_dec_aco_prm")
    
    # Retrieve processing dates for each system
    try:
        query_dat_arrete: str = f"""
            SELECT 
                ID_SYS_GEST,
                DAT_ARRETE,
            FROM `{project_id}.{lib_dec_aco_suivi}.ACO_SUIVI_DAT_ARRETE`
            WHERE ID_TRT = {MV_ID_TRT}
        """
        df_dat_arrete: pd.DataFrame = client.query(query_dat_arrete).to_dataframe()
        
        # Store dates in params with dynamic keys
        for _, row in df_dat_arrete.iterrows():
            id_sys_gest: str = row['ID_SYS_GEST']
            dat_arrete: Any = row['DAT_ARRETE']
            params[f"MV_DAT_ARRETE_{id_sys_gest}"] = dat_arrete
    except Exception as e:
        logger.error(f"Error retrieving ACO_SUIVI_DAT_ARRETE: {str(e)}")
    
    # Retrieve ALTO processing date
    try:
        query_dat_arrete_alto: str = f"""
            SELECT 
                DAT_ARRETE,
            FROM `{project_id}.{lib_dec_aco_suivi}.ACO_SUIVI_DAT_ARRETE_ALTO`
            WHERE ID_TRT = {MV_ID_TRT}
            LIMIT 1
        """
        df_dat_arrete_alto: pd.DataFrame = client.query(query_dat_arrete_alto).to_dataframe()
        
        if not df_dat_arrete_alto.empty:
            DT_ARRETE_ALTO: Any = df_dat_arrete_alto.iloc[0]['DAT_ARRETE']
            params["DT_ARRETE_ALTO"] = DT_ARRETE_ALTO
    except Exception as e:
        logger.error(f"Error retrieving ACO_SUIVI_DAT_ARRETE_ALTO: {str(e)}")
        raise
    
    # Initialize duplicate flag
    FLG_DBL: str = "N"
    params["FLG_DBL"] = FLG_DBL
    
    # Get current datetime
    today_dttm = dt.datetime.now()
    today_dt = dt.datetime.now().date()
    
    # Log start of ODS loading
    try:
        util_dec_aco_alim_suivi_trt(
            ID_TRT=MV_ID_TRT,
            DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
            NOM_TRT="CHARGEMENT_ODS",
            DAT_REEL_TRT=today_dttm,
            MESG="Début Traitement CHARGEMENT_ODS",
            CD_RETOUR=0,
        )
    except Exception as e:
        logger.error(f"Error logging start of treatment: {str(e)}")
        raise
    
    # Get list of tables to load into ODS
    try:
        query_tables_to_load = f"""
            CREATE OR REPLACE TABLE `{project_id}.WORK._A_CHARGER_DANS_ODS` AS
            SELECT  DISTINCT
                B.NOM_TAB_TAMPON,
                B.LIBNAME_TAMPON,
                B.NOM_TAB_ODS,
                B.LIBNAME_ODS,
                B.LIBNAME_ODS_CONF,
                C.ID_TRT,
                C.ID_FIC,
                C.NOM_FIC,
                C.ID_SYS_GEST,
                D.REP_ENT_ACO,
                D.REP_FIC_KO,
                B.TYP_STRUCTURE_FIC,
            FROM `{project_id}.{lib_dec_aco_prm}.ACO_TAB_ODS` B
            INNER JOIN `{project_id}.{lib_dec_aco_suivi}.ACO_SUIVI_CHG_TAMPON_ODS` C
                ON B.NOM_TAB_TAMPON = C.NOM_TAB_TAMPON
            INNER JOIN `{project_id}.{lib_dec_aco_prm}.ACO_SYS_GEST` D
                ON C.ID_SYS_GEST = D.ID_SYS_GEST
            WHERE C.ID_TRT = {MV_ID_TRT}
                AND C.FLG_TRAITE_TAMPON = 'O'
                AND C.FLG_TRAITE_ODS = 'N'
                AND C.DAT_REEL_TRT_ODS IS NULL
        """
        client.query(query_tables_to_load)
    except Exception as e:
        logger.error(f"Error creating _A_CHARGER_DANS_ODS table: {str(e)}")
        raise
    
    # Get number of tables to load
    try:
        query_count = f"""
            SELECT COUNT(*) as NB_TAB_A_CHG
            FROM `{project_id}.WORK._A_CHARGER_DANS_ODS`
        """
        df_count = client.query(query_count).to_dataframe()
        NB_TAB_A_CHG = int(df_count.iloc[0]['NB_TAB_A_CHG'])
    except Exception as e:
        logger.error(f"Error counting tables to load: {str(e)}")
        raise
    
    # Log number of tables to load
    today_dttm = dt.datetime.now()
    try:
        util_dec_aco_alim_suivi_trt(
            ID_TRT=MV_ID_TRT,
            DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
            NOM_TRT="CHARGEMENT_ODS",
            DAT_REEL_TRT=today_dttm,
            MESG=f"{NB_TAB_A_CHG} table(s) à charger dans ODS",
            CD_RETOUR=0,
        )
    except Exception as e:
        logger.error(f"Error logging table count: {str(e)}")
        raise
    
    # Process tables if any exist
    if NB_TAB_A_CHG > 0:
        # Load table information into dataframe
        try:
            query_table_info = f"""
                SELECT 
                    NOM_TAB_TAMPON,
                    NOM_TAB_ODS,
                    TYP_STRUCTURE_FIC,
                    LIBNAME_TAMPON,
                    LIBNAME_ODS,
                    LIBNAME_ODS_CONF,
                    REP_ENT_ACO,
                    REP_FIC_KO,
                    ID_FIC,
                    NOM_FIC,
                    ID_SYS_GEST,
                FROM `{project_id}.WORK._A_CHARGER_DANS_ODS`
            """
            df_tables = client.query(query_table_info).to_dataframe()
        except Exception as e:
            logger.error(f"Error loading table information: {str(e)}")
            raise
        
        # Create sorted structure table
        try:
            query_struct = f"""
                CREATE OR REPLACE TABLE `{project_id}.WORK.STRUCT_TAB_TRIEE_ODS` AS
                SELECT *
                FROM `{project_id}.{lib_dec_aco_prm}.ACO_STRUCT_TAB`
                WHERE FLG_ACTIF = 'O'
                ORDER BY TYP_STRUCTURE_FIC, POSITION_CHAMP
            """
            client.query(query_struct)
        except Exception as e:
            logger.error(f"Error creating sorted structure table: {str(e)}")
            raise
        
        # Process each table
        for i in range(len(df_tables)):
            table_info = df_tables.iloc[i]
            
            NOM_TAB_TAMPON = table_info['NOM_TAB_TAMPON']
            NOM_TAB_ODS = table_info['NOM_TAB_ODS']
            TYP_TAB_ODS = table_info['TYP_STRUCTURE_FIC']
            NOM_LIB_TAMPON = table_info['LIBNAME_TAMPON']
            NOM_LIB_ODS = table_info['LIBNAME_ODS']
            NOM_LIB_ODS_C = table_info['LIBNAME_ODS_CONF']
            REF_FIC_ENT = table_info['REP_ENT_ACO']
            REF_FIC_KO = table_info['REP_FIC_KO']
            ID_FIC = table_info['ID_FIC']
            NOM_FIC = table_info['NOM_FIC']
            ID_SYS_GEST = table_info['ID_SYS_GEST']
            
            logger.info(f"Processing table {i+1}/{NB_TAB_A_CHG}: {NOM_TAB_TAMPON}")
            
            # Initialize exclusion flag
            PRESENCE_VAR_EXCLURE = "N"
            
            # Get row count of staging table
            try:
                query_count_obs = f"""
                    SELECT COUNT(*) as NB_OBS
                    FROM `{project_id}.{NOM_LIB_TAMPON}.{NOM_TAB_TAMPON}`
                """
                df_count_obs = client.query(query_count_obs).to_dataframe()
                NB_OBS = int(df_count_obs.iloc[0]['NB_OBS'])
            except Exception as e:
                logger.error(f"Error counting observations in {NOM_TAB_TAMPON}: {str(e)}")
                raise
            
            # Get structure information for this table type
            try:
                query_structure = f"""
                    SELECT *
                    FROM `{project_id}.WORK.STRUCT_TAB_TRIEE_ODS`
                    WHERE TYP_STRUCTURE_FIC = '{TYP_TAB_ODS}'
                    ORDER BY POSITION_CHAMP
                """
                df_structure = client.query(query_structure).to_dataframe()
            except Exception as e:
                logger.error(f"Error loading structure for {TYP_TAB_ODS}: {str(e)}")
                raise
            
            # Process structure information
            field_info = {}
            NB_CLE = 0
            NB_CLE_C = 0
            NB_CLE_N = 0
            NB_VAR_C = 0
            NB_VAR_N = 0
            NB_VAR_D = 0
            
            for idx, row in df_structure.iterrows():
                field_name = row['NOM_CHAMP']
                field_type = row['TYP_CHAMP']
                field_length = row['LONG_CHAMP']
                field_format = row['FORM_CHAMP']
                field_informat = row['INFORM_CHAMP'] if pd.notna(row['INFORM_CHAMP']) else row['FORM_CHAMP']
                flg_cle = row['FLG_CLE']
                flg_exclusion = row['FLG_EXCLUSION_DIFF']
                flg_conf = row['FLG_CONFIDENTIEL']
                niv_conf = row['NIV_CONFIDENTIEL']
                flg_all_lib = row['FLG_ALL_LIB']
                fmt_decimal = row['FMT_DECIMAL']
                symbol_mt_negatif = row['SYMBOL_MT_NEGATIF']
                
                # Count variables by type
                if flg_cle == 'O':
                    NB_CLE += 1
                    if field_type == 'C':
                        NB_CLE_C += 1
                    else:
                        NB_CLE_N += 1
                
                if field_type == 'C':
                    NB_VAR_C += 1
                elif field_type == 'N':
                    NB_VAR_N += 1
                elif field_type == 'D':
                    NB_VAR_D += 1
                
                # Check for exclusion flag
                if flg_exclusion == 'O':
                    PRESENCE_VAR_EXCLURE = 'O'
                
                # Store field information
                field_info[field_name] = {
                    'type': field_type,
                    'length': field_length,
                    'format': field_format,
                    'informat': field_informat,
                    'flg_cle': flg_cle,
                    'flg_exclusion': flg_exclusion,
                    'flg_conf': flg_conf,
                    'niv_conf': niv_conf,
                    'flg_all_lib': flg_all_lib,
                    'fmt_decimal': fmt_decimal,
                    'symbol_mt_negatif': symbol_mt_negatif,
                }
            
            NB_CHAMP = len(field_info)
            
            # Process only if table has keys
            if NB_CLE > 0:
                # Get key columns
                key_columns = [name for name, info in field_info.items() if info['flg_cle'] == 'O']
                ALL_CLE = ', '.join(key_columns)
                ONE_CLE = key_columns[0] if key_columns else None
                
                logger.info(f"Table {NOM_TAB_TAMPON} has {NB_CLE} key columns: {ALL_CLE}")
                
                # Remove duplicates based on keys
                try:
                    query_dedupe = f"""
                        CREATE OR REPLACE TABLE `{project_id}.WORK.{NOM_TAB_TAMPON}` AS
                        SELECT * EXCEPT(rn) FROM (
                        SELECT *,
                                ROW_NUMBER() OVER (PARTITION BY {ALL_CLE} ORDER BY {ALL_CLE}) AS rn
                        FROM `{project_id}.{NOM_LIB_TAMPON}.{NOM_TAB_TAMPON}` 
                        )
                        WHERE rn = 1
                    """
                    client.query(query_dedupe)
                except Exception as e:
                    logger.error(f"Error deduplicating table {NOM_TAB_TAMPON}: {str(e)}")
                    raise
                
                # Identify duplicates
                try:
                    query_duplicates = f"""
                        CREATE OR REPLACE TABLE `{project_id}.WORK.DBL_{i}` AS
                        SELECT * EXCEPT(rn) FROM (
                        SELECT *,
                                ROW_NUMBER() OVER (PARTITION BY {ALL_CLE} ORDER BY {ALL_CLE}) AS rn
                        FROM `{project_id}.{NOM_LIB_TAMPON}.{NOM_TAB_TAMPON}` 
                        )
                        WHERE rn > 1
                    """
                    client.query(query_duplicates)
                except Exception as e:
                    logger.error(f"Error identifying duplicates in {NOM_TAB_TAMPON}: {str(e)}")
                    raise
                
                # Count duplicates
                try:
                    query_count_dupl = f"""
                        SELECT COUNT(*) as NB_DOUBL
                        FROM `{project_id}.WORK.DBL_{i}`
                    """
                    df_count_dupl = client.query(query_count_dupl).to_dataframe()
                    NB_DOUBL = int(df_count_dupl.iloc[0]['NB_DOUBL'])
                except Exception as e:
                    logger.error(f"Error counting duplicates: {str(e)}")
                    raise
                
                if NB_DOUBL > 0:
                    FLG_DBL = "O"
                    params["FLG_DBL"] = FLG_DBL
                    
                    # Log duplicates
                    today_dttm = dt.datetime.now()
                    try:
                        query_insert_dupl = f"""
                            INSERT INTO `{project_id}.{lib_dec_aco_suivi}.ACO_SUIVI_DOUBLONS`
                            (ID_TRT, DAT_BATCH_TRT, DAT_REEL_TRT, TABLE, CLE, NB_DOUBLONS, NB_OBS_TABLE)
                            VALUES ({MV_ID_TRT}, '{MV_DAT_BATCH_TRT_DATE}', TIMESTAMP('{today_dttm}'), 
                                    '{NOM_TAB_TAMPON}', '{ALL_CLE}', {NB_DOUBL}, {NB_OBS})
                        """
                        client.query(query_insert_dupl)
                    except Exception as e:
                        logger.error(f"Error logging duplicates: {str(e)}")
                        raise
                    
                    # Log duplicate message
                    try:
                        util_dec_aco_alim_suivi_trt(
                            ID_TRT=MV_ID_TRT,
                            DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                            NOM_TRT="CHARGEMENT_ODS",
                            DAT_REEL_TRT=today_dttm,
                            MESG=f"La table {NOM_TAB_TAMPON} contient {NB_DOUBL} doublons",
                            CD_RETOUR=0
                        )
                    except Exception as e:
                        logger.error(f"Error logging duplicate message: {str(e)}")
                        raise
                
                # Perform data validation and type conversion using pandas
                try:
                    # Load staging table to pandas for complex processing
                    query_load_tampon = f"""
                        SELECT *
                        FROM `{project_id}.WORK.{NOM_TAB_TAMPON}`
                    """
                    df_tampon = client.query(query_load_tampon).to_dataframe()
                except Exception as e:
                    logger.error(f"Error loading staging table to pandas: {str(e)}")
                    raise
                
                # Process data validation and conversion
                df_ods, df_erreur, df_err_char, df_err_date, df_err_num, df_err_cle = process_table_validation(
                    df_tampon, field_info, NB_VAR_C, NB_VAR_D, NB_VAR_N, NB_CLE_C, NB_CLE_N
                )
                
                NB_ERR = len(df_erreur)
                
                # Process only if no errors
                if NB_ERR == 0:
                    logger.info(f"Table {NOM_TAB_TAMPON} has no errors, proceeding to load ODS")
                    
                    # Delete temporary staging table
                    try:
                        client.delete_table("WORK", NOM_TAB_TAMPON, project_id)
                    except Exception as e:
                        logger.warning(f"Error deleting temporary table {NOM_TAB_TAMPON}: {str(e)}")
                    
                    # Check if ODS table exists
                    table_exists = client.table_exists(NOM_LIB_ODS, NOM_TAB_ODS)
                    
                    if not table_exists:
                        logger.info(f"Creating new ODS table: {NOM_LIB_ODS}.{NOM_TAB_ODS}")
                        
                        # Add versioning columns
                        today_dttm = dt.datetime.now()
                        
                        # Determine DAT_DEB_VAL based on system
                        if ID_SYS_GEST == "GFP":
                            DAT_DEB_VAL = params.get(f"MV_DAT_ARRETE_{ID_SYS_GEST}")
                        elif ID_SYS_GEST == "ALTO":
                            DAT_DEB_VAL = params.get("DT_ARRETE_ALTO")
                        else:
                            DAT_DEB_VAL = MV_DAT_BATCH_TRT
                        
                        df_ods['DAT_TRT'] = today_dttm
                        df_ods['DAT_DEB_VAL'] = DAT_DEB_VAL
                        df_ods['DAT_FIN_VAL'] = pd.to_datetime('9999-12-31')
                        df_ods['COD_ETA'] = 1
                        df_ods['NUM_SEQ'] = 1
                        df_ods['DAT_DERN_RECPT'] = DAT_DEB_VAL
                        
                        # Select columns for main ODS table (non-confidential or key columns)
                        cols_main = ['DAT_TRT', 'DAT_DEB_VAL', 'DAT_FIN_VAL', 'COD_ETA', 'NUM_SEQ', 'DAT_DERN_RECPT']
                        for field_name, info in field_info.items():
                            if info['flg_cle'] == 'O' or info['flg_conf'] == 'N' or info['flg_all_lib'] == 'O':
                                cols_main.append(field_name)
                        
                        df_ods_main = df_ods[cols_main]
                        
                        # Upload main ODS table
                        try:
                            client.upload_df_to_bq(
                                df=df_ods_main,
                                dataset=NOM_LIB_ODS,
                                table_name=NOM_TAB_ODS,
                                write_disposition='WRITE_TRUNCATE',
                            )
                        except Exception as e:
                            logger.error(f"Error uploading main ODS table: {str(e)}")
                            raise
                        
                        # Create confidential tables if needed
                        if NOM_LIB_ODS_C and NOM_LIB_ODS_C.strip():
                            SIZE_CLE = sum(1 for info in field_info.values() if info['flg_cle'] == 'O')
                            SIZE_MIN_BASE = SIZE_CLE + 6
                            
                            for conf_level in range(1, 5):
                                # Select columns for this confidentiality level
                                cols_conf = ['DAT_TRT', 'DAT_DEB_VAL', 'DAT_FIN_VAL', 'COD_ETA', 'NUM_SEQ', 'DAT_DERN_RECPT']
                                for field_name, info in field_info.items():
                                    if (info['flg_cle'] == 'O' or 
                                        info['flg_all_lib'] == 'O' or 
                                        (info['flg_conf'] == 'O' and info['niv_conf'] == conf_level)):
                                        cols_conf.append(field_name)
                                
                                # Create table first
                                df_ods_conf = df_ods[cols_conf]
                                
                                try:
                                    client.upload_df_to_bq(
                                        df=df_ods_conf,
                                        dataset=f"{NOM_LIB_ODS_C}.{conf_level}",
                                        table_name=NOM_TAB_ODS,
                                        write_disposition='WRITE_TRUNCATE',
                                    )
                                    
                                    # Count columns in created table
                                    try:
                                        query_count_cols = f"""
                                            SELECT COUNT(*) as NB_COL
                                            FROM `{project_id}.{NOM_LIB_ODS_C}.{conf_level}.INFORMATION_SCHEMA.COLUMNS`
                                            WHERE table_name = '{NOM_TAB_ODS}'
                                        """
                                        df_col_count = client.query(query_count_cols).to_dataframe()
                                        NB_COL_TABLI = int(df_col_count.iloc[0]['NB_COL'])
                                        
                                        # Delete table if it only contains base columns (no confidential data)
                                        if NB_COL_TABLI == SIZE_MIN_BASE:
                                            logger.info(f"Deleting confidential table {NOM_LIB_ODS_C}.{conf_level}.{NOM_TAB_ODS} - contains no confidential data")
                                            try:
                                                client.delete_table(f"{NOM_LIB_ODS_C}.{conf_level}", NOM_TAB_ODS, project_id)
                                            except Exception as e:
                                                logger.warning(f"Error deleting empty confidential table: {str(e)}")
                                    except Exception as e:
                                        logger.warning(f"Error counting columns in confidential table level {conf_level}: {str(e)}")
                                        
                                except Exception as e:
                                    logger.warning(f"Error creating confidential table level {conf_level}: {str(e)}")
                    
                    else:
                        # Handle case when table already exists - implement historization
                        logger.info(f"ODS table {NOM_LIB_ODS}.{NOM_TAB_ODS} already exists, performing historization")
                        
                        # Determine DAT_DEB_VAL based on system
                        if ID_SYS_GEST == "GFP":
                            DAT_DEB_VAL = params.get(f"MV_DAT_ARRETE_{ID_SYS_GEST}")
                        elif ID_SYS_GEST == "ALTO":
                            DAT_DEB_VAL = params.get("DT_ARRETE_ALTO")
                        else:
                            DAT_DEB_VAL = MV_DAT_BATCH_TRT
                        
                        today_dttm = dt.datetime.now()
                        
                        # Get list of key columns
                        key_columns = [name for name, info in field_info.items() if info['flg_cle'] == 'O']
                        
                        # Get list of comparable columns (exclude fields marked for exclusion from diff)
                        comparable_columns = [name for name, info in field_info.items() 
                                            if info.get('flg_exclusion') != 'O' and info['flg_cle'] != 'O']
                        
                        # Get all data columns (for selection)
                        all_data_columns = list(field_info.keys())
                        
                        # Load existing ODS data (only active records)
                        try:
                            query_existing = f"""
                                SELECT *
                                FROM `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}`
                                WHERE COD_ETA = 1
                            """
                            df_existing = client.query(query_existing).to_dataframe()
                        except Exception as e:
                            logger.error(f"Error loading existing ODS data: {str(e)}")
                            raise
                        
                        # Prepare new data with versioning columns
                        df_ods['DAT_TRT'] = today_dttm
                        df_ods['DAT_DEB_VAL'] = DAT_DEB_VAL
                        df_ods['DAT_FIN_VAL'] = pd.to_datetime('9999-12-31')
                        df_ods['COD_ETA'] = 1
                        df_ods['DAT_DERN_RECPT'] = DAT_DEB_VAL
                        
                        # Identify records to update and insert
                        # Merge on keys to find matching records
                        df_merged = df_existing.merge(
                            df_ods,
                            on=key_columns,
                            how='outer',
                            suffixes=('_old', '_new'),
                            indicator=True
                        )
                        
                        # Records only in existing data (deleted) - close them out
                        df_deleted = df_merged[df_merged['_merge'] == 'left_only'].copy()
                        
                        # Records only in new data (inserted) - add them
                        df_inserted = df_merged[df_merged['_merge'] == 'right_only'].copy()
                        
                        # Records in both - check if they changed
                        df_both = df_merged[df_merged['_merge'] == 'both'].copy()
                        
                        # Compare comparable columns to detect changes
                        changed_mask = pd.Series([False] * len(df_both))
                        for col in comparable_columns:
                            col_old = f"{col}_old"
                            col_new = f"{col}_new"
                            if col_old in df_both.columns and col_new in df_both.columns:
                                # Handle NaN comparisons properly
                                changed_mask |= (
                                    (df_both[col_old].fillna('__NULL__') != df_both[col_new].fillna('__NULL__'))
                                )
                        
                        df_changed = df_both[changed_mask].copy()
                        df_unchanged = df_both[~changed_mask].copy()
                        
                        # Update DAT_FIN_VAL for deleted and changed records
                        records_to_close = []
                        
                        # Collect IDs of records to close (deleted records)
                        for _, row in df_deleted.iterrows():
                            record_keys = {k: row[k] for k in key_columns}
                            records_to_close.append(record_keys)
                        
                        # Collect IDs of records to close (changed records)
                        for _, row in df_changed.iterrows():
                            record_keys = {k: row[f"{k}_old"] for k in key_columns}
                            records_to_close.append(record_keys)
                        
                        # Close out old records by updating DAT_FIN_VAL
                        if records_to_close:
                            for record_keys in records_to_close:
                                where_clause = ' AND '.join([f"{k} = '{v}'" if isinstance(v, str) else f"{k} = {v}" 
                                                            for k, v in record_keys.items()])
                                try:
                                    query_update = f"""
                                        UPDATE `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}`
                                        SET DAT_FIN_VAL = DATE('{DAT_DEB_VAL}'),
                                            COD_ETA = 0
                                        WHERE {where_clause}
                                            AND COD_ETA = 1
                                    """
                                    client.query(query_update)
                                except Exception as e:
                                    logger.error(f"Error updating DAT_FIN_VAL for record: {str(e)}")
                                    raise
                        
                        # Prepare records to insert (new and changed)
                        df_to_insert = pd.DataFrame()
                        
                        # Add new records
                        if len(df_inserted) > 0:
                            for col in all_data_columns:
                                col_new = f"{col}_new"
                                if col_new in df_inserted.columns:
                                    df_to_insert[col] = df_inserted[col_new]
                            df_to_insert['NUM_SEQ'] = 1
                        
                        # Add changed records with incremented NUM_SEQ
                        if len(df_changed) > 0:
                            df_changed_insert = pd.DataFrame()
                            for col in all_data_columns:
                                col_new = f"{col}_new"
                                if col_new in df_changed.columns:
                                    df_changed_insert[col] = df_changed[col_new]
                            
                            # Get max NUM_SEQ for each key and increment
                            for idx, row in df_changed.iterrows():
                                record_keys = {k: row[f"{k}_old"] for k in key_columns}
                                where_clause = ' AND '.join([f"{k} = '{v}'" if isinstance(v, str) else f"{k} = {v}" 
                                                            for k, v in record_keys.items()])
                                try:
                                    query_max_seq = f"""
                                        SELECT COALESCE(MAX(NUM_SEQ), 0) as MAX_SEQ
                                        FROM `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}`
                                        WHERE {where_clause}
                                    """
                                    df_max_seq = client.query(query_max_seq).to_dataframe()
                                    max_seq = int(df_max_seq.iloc[0]['MAX_SEQ'])
                                    df_changed_insert.loc[idx, 'NUM_SEQ'] = max_seq + 1
                                except Exception as e:
                                    logger.error(f"Error getting max NUM_SEQ: {str(e)}")
                                    df_changed_insert.loc[idx, 'NUM_SEQ'] = 1
                            
                            df_to_insert = pd.concat([df_to_insert, df_changed_insert], ignore_index=True)
                        
                        # Add versioning columns to records to insert
                        if len(df_to_insert) > 0:
                            df_to_insert['DAT_TRT'] = today_dttm
                            df_to_insert['DAT_DEB_VAL'] = DAT_DEB_VAL
                            df_to_insert['DAT_FIN_VAL'] = pd.to_datetime('9999-12-31')
                            df_to_insert['COD_ETA'] = 1
                            df_to_insert['DAT_DERN_RECPT'] = DAT_DEB_VAL
                            
                            # Select columns for main ODS table
                            cols_main = ['DAT_TRT', 'DAT_DEB_VAL', 'DAT_FIN_VAL', 'COD_ETA', 'NUM_SEQ', 'DAT_DERN_RECPT']
                            for field_name, info in field_info.items():
                                if info['flg_cle'] == 'O' or info['flg_conf'] == 'N' or info['flg_all_lib'] == 'O':
                                    cols_main.append(field_name)
                            
                            df_to_insert_main = df_to_insert[cols_main]
                            
                            # Insert new records into main ODS table
                            try:
                                client.upload_df_to_bq(
                                    df=df_to_insert_main,
                                    dataset=NOM_LIB_ODS,
                                    table_name=NOM_TAB_ODS,
                                    write_disposition='WRITE_APPEND',
                                )
                                logger.info(f"Inserted {len(df_to_insert_main)} records into {NOM_LIB_ODS}.{NOM_TAB_ODS}")
                            except Exception as e:
                                logger.error(f"Error inserting records into main ODS table: {str(e)}")
                                raise
                            
                            # Handle confidential tables
                            if NOM_LIB_ODS_C and NOM_LIB_ODS_C.strip():
                                SIZE_CLE = sum(1 for info in field_info.values() if info['flg_cle'] == 'O')
                                SIZE_MIN_BASE = SIZE_CLE + 6
                                
                                for conf_level in range(1, 5):
                                    # Select columns for this confidentiality level
                                    cols_conf = ['DAT_TRT', 'DAT_DEB_VAL', 'DAT_FIN_VAL', 'COD_ETA', 'NUM_SEQ', 'DAT_DERN_RECPT']
                                    for field_name, info in field_info.items():
                                        if (info['flg_cle'] == 'O' or 
                                            info['flg_all_lib'] == 'O' or 
                                            (info['flg_conf'] == 'O' and info['niv_conf'] == conf_level)):
                                            cols_conf.append(field_name)
                                    
                                    # Only process if there are confidential columns for this level
                                    if len(cols_conf) > SIZE_MIN_BASE:
                                        df_to_insert_conf = df_to_insert[cols_conf]
                                        
                                        # Check if confidential table exists
                                        conf_table_exists = client.table_exists(f"{NOM_LIB_ODS_C}.{conf_level}", NOM_TAB_ODS)
                                        
                                        if conf_table_exists:
                                            # Update existing confidential records (close them out)
                                            if records_to_close:
                                                for record_keys in records_to_close:
                                                    where_clause = ' AND '.join([f"{k} = '{v}'" if isinstance(v, str) else f"{k} = {v}" 
                                                                                for k, v in record_keys.items()])
                                                    try:
                                                        query_update_conf = f"""
                                                            UPDATE `{project_id}.{NOM_LIB_ODS_C}.{conf_level}.{NOM_TAB_ODS}`
                                                            SET DAT_FIN_VAL = DATE('{DAT_DEB_VAL}'),
                                                                COD_ETA = 0
                                                            WHERE {where_clause}
                                                                AND COD_ETA = 1
                                                        """
                                                        client.query(query_update_conf)
                                                    except Exception as e:
                                                        logger.warning(f"Error updating confidential table level {conf_level}: {str(e)}")
                                            
                                            # Insert new records
                                            try:
                                                client.upload_df_to_bq(
                                                    df=df_to_insert_conf,
                                                    dataset=f"{NOM_LIB_ODS_C}.{conf_level}",
                                                    table_name=NOM_TAB_ODS,
                                                    write_disposition='WRITE_APPEND'
                                                )
                                            except Exception as e:
                                                logger.warning(f"Error inserting into confidential table level {conf_level}: {str(e)}")
                                        else:
                                            # Create new confidential table
                                            try:
                                                client.upload_df_to_bq(
                                                    df=df_to_insert_conf,
                                                    dataset=f"{NOM_LIB_ODS_C}.{conf_level}",
                                                    table_name=NOM_TAB_ODS,
                                                    write_disposition='WRITE_TRUNCATE'
                                                )
                                            except Exception as e:
                                                logger.warning(f"Error creating confidential table level {conf_level}: {str(e)}")
                        
                        logger.info(f"Historization completed for {NOM_TAB_ODS}: {len(df_inserted)} new, {len(df_changed)} changed, {len(df_deleted)} deleted")
                    
                    # Update tracking table to mark as processed (applies to both new and existing tables)
                    today_dttm = dt.datetime.now()
                    try:
                        query_update_tracking = f"""
                            UPDATE `{project_id}.{lib_dec_aco_suivi}.ACO_SUIVI_CHG_TAMPON_ODS`
                            SET FLG_TRAITE_ODS = 'O',
                                DAT_REEL_TRT_ODS = TIMESTAMP('{today_dttm}')
                            WHERE ID_TRT = {MV_ID_TRT}
                                AND NOM_TAB_TAMPON = '{NOM_TAB_TAMPON}'
                        """
                        client.query(query_update_tracking)
                        logger.info(f"Successfully updated tracking for table {NOM_TAB_TAMPON}")
                    except Exception as e:
                        logger.error(f"Error updating tracking table for {NOM_TAB_TAMPON}: {str(e)}")
                
                else:
                    # Handle validation errors
                    logger.error(f"Table {NOM_TAB_TAMPON} has {NB_ERR} validation errors, skipping ODS load")
                    
                    # Upload error records to tracking tables
                    today_dttm = dt.datetime.now()
                    
                    # Upload character errors
                    if len(df_err_char) > 0:
                        try:
                            df_err_char['ID_TRT'] = MV_ID_TRT
                            df_err_char['DAT_BATCH_TRT'] = MV_DAT_BATCH_TRT
                            df_err_char['DAT_REEL_TRT'] = today_dttm
                            df_err_char['NOM_TAB'] = NOM_TAB_TAMPON
                            client.upload_df_to_bq(
                                df=df_err_char,
                                dataset=lib_dec_aco_suivi,
                                table_name='ACO_ERREURS_CHAR',
                                write_disposition='WRITE_APPEND',
                            )
                        except Exception as e:
                            logger.warning(f"Error uploading character errors: {str(e)}")
                    
                    # Upload date errors
                    if len(df_err_date) > 0:
                        try:
                            df_err_date['ID_TRT'] = MV_ID_TRT
                            df_err_date['DAT_BATCH_TRT'] = MV_DAT_BATCH_TRT
                            df_err_date['DAT_REEL_TRT'] = today_dttm
                            df_err_date['NOM_TAB'] = NOM_TAB_TAMPON
                            client.upload_df_to_bq(
                                df=df_err_date,
                                dataset=lib_dec_aco_suivi,
                                table_name='ACO_ERREURS_DATE',
                                write_disposition='WRITE_APPEND',
                            )
                        except Exception as e:
                            logger.warning(f"Error uploading date errors: {str(e)}")
                    
                    # Upload numeric errors
                    if len(df_err_num) > 0:
                        try:
                            df_err_num['ID_TRT'] = MV_ID_TRT
                            df_err_num['DAT_BATCH_TRT'] = MV_DAT_BATCH_TRT
                            df_err_num['DAT_REEL_TRT'] = today_dttm
                            df_err_num['NOM_TAB'] = NOM_TAB_TAMPON
                            client.upload_df_to_bq(
                                df=df_err_num,
                                dataset=lib_dec_aco_suivi,
                                table_name='ACO_ERREURS_NUM',
                                write_disposition='WRITE_APPEND',
                            )
                        except Exception as e:
                            logger.warning(f"Error uploading numeric errors: {str(e)}")
                    
                    # Upload key errors
                    if len(df_err_cle) > 0:
                        try:
                            df_err_cle['ID_TRT'] = MV_ID_TRT
                            df_err_cle['DAT_BATCH_TRT'] = MV_DAT_BATCH_TRT
                            df_err_cle['DAT_REEL_TRT'] = today_dttm
                            df_err_cle['NOM_TAB'] = NOM_TAB_TAMPON
                            client.upload_df_to_bq(
                                df=df_err_cle,
                                dataset=lib_dec_aco_suivi,
                                table_name='ACO_ERREURS_CLE',
                                write_disposition='WRITE_APPEND',
                            )
                        except Exception as e:
                            logger.warning(f"Error uploading key errors: {str(e)}")
                    
                    # Log error summary
                    try:
                        util_dec_aco_alim_suivi_trt(
                            ID_TRT=MV_ID_TRT,
                            DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                            NOM_TRT="CHARGEMENT_ODS",
                            DAT_REEL_TRT=today_dttm,
                            MESG=f"Table {NOM_TAB_TAMPON} has {NB_ERR} validation errors",
                            CD_RETOUR=1
                        )
                    except Exception as e:
                        logger.error(f"Error logging validation errors: {str(e)}")
                
                # Cleanup temporary work tables
                try:
                    temp_tables = [f"DBL_{i}", "STRUCTURE_TABLE", "CLE"]
                    for temp_table in temp_tables:
                        if client.table_exists("WORK", temp_table):
                            client.delete_table("WORK", temp_table, project_id)
                except Exception as e:
                    logger.warning(f"Error cleaning up temporary tables: {str(e)}")

                        
                        # Count new keys inserted
                try:
                                query_count_new = f"""
                                    SELECT COUNT(*) as NB_NEW_CLE
                                    FROM `{project_id}.WORK.__{NOM_TAB_ODS}_{i}`
                                """
                                df_count_new = client.query(query_count_new).to_dataframe()
                                NB_NEW_CLE = int(df_count_new.iloc[0]['NB_NEW_CLE'])
                except Exception as e:
                    logger.error(f"Error counting new keys: {str(e)}")
                    raise
                        
                    # Log successful load
                today_dttm = dt.datetime.now()
                try:
                    util_dec_aco_alim_suivi_trt(
                        ID_TRT=MV_ID_TRT,
                        DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                        NOM_TRT="CHARGEMENT_ODS",
                        DAT_REEL_TRT=today_dttm,
                        MESG=f"La table {NOM_TAB_TAMPON} est chargé dans ODS : {NB_NEW_CLE} NOUVELLES CLES.",
                        CD_RETOUR=0
                    )
                except Exception as e:
                    logger.error(f"Error logging table load: {str(e)}")
                    raise
                    
                    # Update tracking table
                try:
                    query_update_tracking = f"""
                        UPDATE `{project_id}.{lib_dec_aco_suivi}.ACO_SUIVI_CHG_TAMPON_ODS`
                        SET FLG_TRAITE_ODS = 'O',
                            CD_STATUT_ODS = 'C',
                            DAT_REEL_TRT_ODS = TIMESTAMP('{today_dttm}')
                        WHERE ID_TRT = {MV_ID_TRT}
                            AND DAT_BATCH_TRT = {MV_DAT_BATCH_TRT}
                            AND ID_FIC = '{ID_FIC}'
                    """
                    client.query(query_update_tracking)
                except Exception as e:
                    logger.error(f"Error updating tracking table: {str(e)}")
                    raise
                    
                # Get MV_EXTENSION from params
                MV_EXTENSION = params.get("MV_EXTENSION", "")
                
                # Move file to archives directory
                try:
                    source_path = f"{REF_FIC_ENT}/ok/{NOM_FIC}"
                    dest_path = f"{REF_FIC_ENT}/archives/{NOM_FIC}.{MV_EXTENSION}"
                    
                    # Use GCS operations if paths are GCS URIs
                    if source_path.startswith("gs://"):
                        # Extract bucket and paths
                        source_parts = source_path.replace("gs://", "").split("/", 1)
                        dest_parts = dest_path.replace("gs://", "").split("/", 1)
                        
                        source_bucket = source_parts[0]
                        source_blob = source_parts[1]
                        dest_bucket = dest_parts[0]
                        dest_blob = dest_parts[1]
                        
                        # Move file in GCS
                        from google.cloud import storage
                        storage_client_gcs = storage.Client(project=project_id)
                        source_bucket_obj = storage_client_gcs.bucket(source_bucket)
                        source_blob_obj = source_bucket_obj.blob(source_blob)
                        dest_bucket_obj = storage_client_gcs.bucket(dest_bucket)
                        
                        source_bucket_obj.copy_blob(source_blob_obj, dest_bucket_obj, dest_blob)
                        source_blob_obj.delete()
                        
                        move_success = True
                    else:
                        # Use local file operations
                        import shutil
                        shutil.move(source_path, dest_path)
                        move_success = True
                except Exception as e:
                    logger.warning(f"Error moving file {NOM_FIC}: {str(e)}")
                    move_success = False
                    
                if move_success:
                    today_dttm = dt.datetime.now()
                    try:
                        util_dec_aco_alim_suivi_trt(
                            ID_TRT=MV_ID_TRT,
                            DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                            NOM_TRT="CHARGEMENT_ODS",
                            DAT_REEL_TRT=today_dttm,
                            MESG=f"Le fichier {NOM_FIC} a été déplacé dans le répertoire {REF_FIC_ENT}/archives.",
                            CD_RETOUR=0
                        )
                    except Exception as e:
                        logger.error(f"Error logging file move: {str(e)}")
                        
                    # Zip the file
                    try:
                        archive_file = f"{NOM_FIC}.{MV_EXTENSION}"
                        archive_path = f"{REF_FIC_ENT}/archives/{archive_file}"
                        zip_path = f"{REF_FIC_ENT}/archives/Archives_{MV_EXTENSION}.zip"
                        
                        if archive_path.startswith("gs://"):
                            # For GCS, download, zip locally, upload
                            import tempfile
                            import zipfile
                            
                            with tempfile.TemporaryDirectory() as tmpdir:
                                # Download file
                                archive_parts = archive_path.replace("gs://", "").split("/", 1)
                                archive_bucket = archive_parts[0]
                                archive_blob = archive_parts[1]
                                
                                local_file = f"{tmpdir}/{archive_file}"
                                storage_client_gcs = storage.Client(project=project_id)
                                bucket_obj = storage_client_gcs.bucket(archive_bucket)
                                blob_obj = bucket_obj.blob(archive_blob)
                                blob_obj.download_to_filename(local_file)
                                
                                # Create or update zip
                                local_zip = f"{tmpdir}/Archives_{MV_EXTENSION}.zip"
                                
                                # Download existing zip if it exists
                                zip_parts = zip_path.replace("gs://", "").split("/", 1)
                                zip_blob = zip_parts[1]
                                zip_blob_obj = bucket_obj.blob(zip_blob)
                                
                                if zip_blob_obj.exists():
                                    zip_blob_obj.download_to_filename(local_zip)
                                    mode = 'a'
                                else:
                                    mode = 'w'
                                
                                # Add file to zip
                                with zipfile.ZipFile(local_zip, mode) as zf:
                                    zf.write(local_file, archive_file)
                                
                                # Upload zip back
                                zip_blob_obj.upload_from_filename(local_zip)
                                
                                zip_success = True
                        else:
                            # Local file system
                            import zipfile
                            
                            mode = 'a' if os.path.exists(zip_path) else 'w'
                            with zipfile.ZipFile(zip_path, mode) as zf:
                                zf.write(archive_path, archive_file)
                            
                            zip_success = True
                    except Exception as e:
                        logger.warning(f"Error zipping file {NOM_FIC}: {str(e)}")
                        zip_success = False
                        
                        if zip_success:
                            today_dttm = dt.datetime.now()
                            try:
                                util_dec_aco_alim_suivi_trt(
                                    ID_TRT=MV_ID_TRT,
                                    DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                                    NOM_TRT="CHARGEMENT_ODS",
                                    DAT_REEL_TRT=today_dttm,
                                    MESG=f"Le fichier {NOM_FIC} a été ajouté à l'archive Archives_{MV_EXTENSION}.zip.",
                                    CD_RETOUR=0
                                )
                            except Exception as e:
                                logger.error(f"Error logging zip success: {str(e)}")
                            
                            # Remove the unzipped file
                            try:
                                if archive_path.startswith("gs://"):
                                    archive_parts = archive_path.replace("gs://", "").split("/", 1)
                                    archive_bucket = archive_parts[0]
                                    archive_blob = archive_parts[1]
                                    
                                    storage_client_gcs = storage.Client(project=project_id)
                                    bucket_obj = storage_client_gcs.bucket(archive_bucket)
                                    blob_obj = bucket_obj.blob(archive_blob)
                                    blob_obj.delete()
                                else:
                                    import os
                                    os.remove(archive_path)
                            except Exception as e:
                                logger.warning(f"Error removing archived file: {str(e)}")
                        else:
                            today_dttm = dt.datetime.now()
                            try:
                                util_dec_aco_alim_suivi_trt(
                                    ID_TRT=MV_ID_TRT,
                                    DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                                    NOM_TRT="CHARGEMENT_ODS",
                                    DAT_REEL_TRT=today_dttm,
                                    MESG=f"Impossible d'ajouter le fichier {NOM_FIC} à l'archive Archives_{MV_EXTENSION}.zip.",
                                    CD_RETOUR=99
                                )
                            except Exception as e:
                                logger.error(f"Error logging zip failure: {str(e)}")
                    else:
                        today_dttm = dt.datetime.now()
                        try:
                            util_dec_aco_alim_suivi_trt(
                                ID_TRT=MV_ID_TRT,
                                DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                                NOM_TRT="CHARGEMENT_ODS",
                                DAT_REEL_TRT=today_dttm,
                                MESG=f"Le fichier {NOM_FIC} n'a pas pu être déplacé dans le répertoire {REF_FIC_ENT}/archives.",
                                CD_RETOUR=0
                            )
                        except Exception as e:
                            logger.error(f"Error logging move failure: {str(e)}")
                
                # ELSE case: ODS table already exists - perform historization
                else:
                    logger.info(f"ODS table {NOM_LIB_ODS}.{NOM_TAB_ODS} already exists, performing historization")
                    
                    # Determine DAT_DEB_VAL based on system
                    if ID_SYS_GEST == "GFP":
                        DAT_DEB_VAL = params.get(f"MV_DAT_ARRETE_{ID_SYS_GEST}")
                    elif ID_SYS_GEST == "ALTO":
                        DAT_DEB_VAL = params.get("DT_ARRETE_ALTO")
                    else:
                        DAT_DEB_VAL = MV_DAT_BATCH_TRT
                    
                    today_dttm = dt.datetime.now()
                    
                    # Sort existing ODS table
                    key_columns = [name for name, info in field_info.items() if info['flg_cle'] == 'O']
                    key_cols_str = ', '.join(key_columns)
                    
                    try:
                        query_sort_ods = f"""
                            CREATE OR REPLACE TABLE `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}_SORTED` AS
                            SELECT *
                            FROM `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}`
                            ORDER BY {key_cols_str}, DAT_DEB_VAL, DAT_FIN_VAL, NUM_SEQ, COD_ETA
                        """
                        client.query(query_sort_ods)
                    except Exception as e:
                        logger.error(f"Error sorting ODS table: {str(e)}")
                        raise
                    
                    # Sort and merge confidential tables if they exist
                    if NOM_LIB_ODS_C and NOM_LIB_ODS_C.strip():
                        for conf_level in range(1, 5):
                            conf_table_exists = client.table_exists(f"{NOM_LIB_ODS_C}.{conf_level}", NOM_TAB_ODS)
                            
                            if conf_table_exists:
                                try:
                                    query_sort_conf = f"""
                                        CREATE OR REPLACE TABLE `{project_id}.WORK.{NOM_TAB_ODS}_{conf_level}` AS
                                        SELECT *
                                        FROM `{project_id}.{NOM_LIB_ODS_C}.{conf_level}.{NOM_TAB_ODS}`
                                        ORDER BY {key_cols_str}, DAT_DEB_VAL, DAT_FIN_VAL, NUM_SEQ, COD_ETA
                                    """
                                    client.query(query_sort_conf)
                                except Exception as e:
                                    logger.warning(f"Error sorting confidential table level {conf_level}: {str(e)}")
                        
                        # Merge all data (main + confidential tables)
                        try:
                            # Build UNION ALL query for all confidential levels
                            union_parts = [f"SELECT * FROM `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}_SORTED`"]
                            
                            for conf_level in range(1, 5):
                                if client.table_exists("WORK", f"{NOM_TAB_ODS}_{conf_level}"):
                                    union_parts.append(f"SELECT * FROM `{project_id}.WORK.{NOM_TAB_ODS}_{conf_level}`")
                            
                            query_merge_all = f"""
                                CREATE OR REPLACE TABLE `{project_id}.WORK._ALL_DONNEES_{i}` AS
                                SELECT *
                                FROM (
                                    {' UNION ALL '.join(union_parts)}
                                )
                                WHERE COD_ETA = 1
                                ORDER BY {key_cols_str}, DAT_DEB_VAL, DAT_FIN_VAL, NUM_SEQ, COD_ETA
                            """
                            client.query(query_merge_all)
                        except Exception as e:
                            logger.error(f"Error merging all data: {str(e)}")
                            raise
                    else:
                        # No confidential tables, just filter active records
                        try:
                            query_active_only = f"""
                                CREATE OR REPLACE TABLE `{project_id}.WORK._ALL_DONNEES_{i}` AS
                                SELECT *
                                FROM `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}_SORTED`
                                WHERE COD_ETA = 1
                                ORDER BY {key_cols_str}, DAT_DEB_VAL, DAT_FIN_VAL, NUM_SEQ, COD_ETA
                            """
                            client.query(query_active_only)
                        except Exception as e:
                            logger.error(f"Error filtering active records: {str(e)}")
                            raise
                    
                    # Sort new data
                    try:
                        query_sort_new = f"""
                            CREATE OR REPLACE TABLE `{project_id}.WORK._{NOM_TAB_ODS}_{i}_SORTED` AS
                            SELECT *
                            FROM `{project_id}.WORK._{NOM_TAB_ODS}_{i}`
                            ORDER BY {key_cols_str}
                        """
                        client.query(query_sort_new)
                    except Exception as e:
                        logger.error(f"Error sorting new data: {str(e)}")
                        raise
                    
                    # Identify new vs existing keys using pandas for complex merge logic
                    try:
                        # Load data to pandas
                        query_load_existing = f"""
                            SELECT *
                            FROM `{project_id}.WORK._ALL_DONNEES_{i}`
                        """
                        df_existing = client.query(query_load_existing).to_dataframe()
                        
                        query_load_new = f"""
                            SELECT *
                            FROM `{project_id}.WORK._{NOM_TAB_ODS}_{i}_SORTED`
                        """
                        df_new = client.query(query_load_new).to_dataframe()
                    except Exception as e:
                        logger.error(f"Error loading data for merge: {str(e)}")
                        raise
                    
                    # Prepare new records with versioning columns
                    df_new['DAT_TRT'] = today_dttm
                    df_new['DAT_DEB_VAL'] = DAT_DEB_VAL
                    df_new['DAT_FIN_VAL'] = pd.to_datetime('9999-12-31')
                    df_new['COD_ETA'] = 1
                    df_new['DAT_DERN_RECPT'] = DAT_DEB_VAL
                    
                    # Merge to identify new vs existing keys
                    df_merged = df_existing.merge(
                        df_new,
                        on=key_columns,
                        how='outer',
                        suffixes=('_old', '_new'),
                        indicator=True
                    )
                    
                    # New keys (only in new data)
                    df_new_keys = df_merged[df_merged['_merge'] == 'right_only'].copy()
                    
                    # Existing keys (in both)
                    df_existing_keys = df_merged[df_merged['_merge'] == 'both'].copy()
                    
                    # Prepare new keys output
                    df_nouvelles_cles = pd.DataFrame()
                    if len(df_new_keys) > 0:
                        for col in field_info.keys():
                            col_new = f"{col}_new"
                            if col_new in df_new_keys.columns:
                                df_nouvelles_cles[col] = df_new_keys[col_new]
                        
                        df_nouvelles_cles['DAT_TRT'] = today_dttm
                        df_nouvelles_cles['DAT_DEB_VAL'] = DAT_DEB_VAL
                        df_nouvelles_cles['DAT_FIN_VAL'] = pd.to_datetime('9999-12-31')
                        df_nouvelles_cles['COD_ETA'] = 1
                        df_nouvelles_cles['NUM_SEQ'] = 1
                        df_nouvelles_cles['DAT_DERN_RECPT'] = DAT_DEB_VAL
                    
                    # Prepare existing keys output (will be split into changed/unchanged later)
                    df_anciennes_cles = pd.DataFrame()
                    if len(df_existing_keys) > 0:
                        for col in field_info.keys():
                            col_new = f"{col}_new"
                            if col_new in df_existing_keys.columns:
                                df_anciennes_cles[col] = df_existing_keys[col_new]
                        
                        df_anciennes_cles['DAT_TRT'] = today_dttm
                        df_anciennes_cles['DAT_DEB_VAL'] = DAT_DEB_VAL
                        df_anciennes_cles['DAT_FIN_VAL'] = pd.to_datetime('9999-12-31')
                        df_anciennes_cles['COD_ETA'] = 1
                        df_anciennes_cles['DAT_DERN_RECPT'] = DAT_DEB_VAL
                    
                    # Upload new keys tables
                    try:
                        # Main table columns
                        cols_main = ['DAT_TRT', 'DAT_DEB_VAL', 'DAT_FIN_VAL', 'COD_ETA', 'NUM_SEQ', 'DAT_DERN_RECPT']
                        for field_name, info in field_info.items():
                            if info['flg_cle'] == 'O' or info['flg_conf'] == 'N' or info['flg_all_lib'] == 'O':
                                cols_main.append(field_name)
                        
                        if len(df_nouvelles_cles) > 0:
                            df_nouvelles_cles_main = df_nouvelles_cles[cols_main]
                            client.upload_df_to_bq(
                                df=df_nouvelles_cles_main,
                                dataset="WORK",
                                table_name=f"_NOUVELLES_CLES{i}",
                                write_disposition='WRITE_TRUNCATE'
                            )
                        else:
                            # Create empty table
                            client.upload_df_to_bq(
                                df=pd.DataFrame(columns=cols_main),
                                dataset="WORK",
                                table_name=f"_NOUVELLES_CLES{i}",
                                write_disposition='WRITE_TRUNCATE'
                            )
                        
                        # Confidential tables
                        for conf_level in range(1, 5):
                            cols_conf = ['DAT_TRT', 'DAT_DEB_VAL', 'DAT_FIN_VAL', 'COD_ETA', 'NUM_SEQ', 'DAT_DERN_RECPT']
                            for field_name, info in field_info.items():
                                if (info['flg_cle'] == 'O' or 
                                    info['flg_all_lib'] == 'O' or 
                                    (info['flg_conf'] == 'O' and info['niv_conf'] == conf_level)):
                                    cols_conf.append(field_name)
                            
                            if len(df_nouvelles_cles) > 0:
                                df_nouvelles_cles_conf = df_nouvelles_cles[cols_conf]
                                client.upload_df_to_bq(
                                    df=df_nouvelles_cles_conf,
                                    dataset="WORK",
                                    table_name=f"_NOUVELLES_CLES_CONF{i}_{conf_level}",
                                    write_disposition='WRITE_TRUNCATE'
                                )
                            else:
                                client.upload_df_to_bq(
                                    df=pd.DataFrame(columns=cols_conf),
                                    dataset="WORK",
                                    table_name=f"_NOUVELLES_CLES_CONF{i}_{conf_level}",
                                    write_disposition='WRITE_TRUNCATE'
                                )
                        
                        # Upload anciennes_cles
                        if len(df_anciennes_cles) > 0:
                            client.upload_df_to_bq(
                                df=df_anciennes_cles,
                                dataset="WORK",
                                table_name=f"_ANCIENNES_CLES{i}",
                                write_disposition='WRITE_TRUNCATE'
                            )
                        else:
                            client.upload_df_to_bq(
                                df=pd.DataFrame(columns=list(field_info.keys()) + ['DAT_TRT', 'DAT_DEB_VAL', 'DAT_FIN_VAL', 'COD_ETA', 'DAT_DERN_RECPT']),
                                dataset="WORK",
                                table_name=f"_ANCIENNES_CLES{i}",
                                write_disposition='WRITE_TRUNCATE'
                            )
                    except Exception as e:
                        logger.error(f"Error uploading new/existing keys tables: {str(e)}")
                        raise
                    
                    # Get NOM_COURT (short name for table type)
                    try:
                        query_nom_court = f"""
                            SELECT TYP_STRUCTURE_FIC
                            FROM `{project_id}.S_DACPRM.ACO_TAB_ODS`
                            WHERE NOM_TAB_TAMPON = '{NOM_TAB_TAMPON}'
                            LIMIT 1
                        """
                        df_nom_court = client.query(query_nom_court).to_dataframe()
                        NOM_COURT = df_nom_court.iloc[0]['TYP_STRUCTURE_FIC'] if not df_nom_court.empty else TYP_TAB_ODS
                    except Exception as e:
                        logger.warning(f"Error getting NOM_COURT: {str(e)}")
                        NOM_COURT = TYP_TAB_ODS
                    
                    # Get all comparable columns (excluding those marked for exclusion from diff)
                    try:
                        query_all_cle_comp = f"""
                            SELECT NOM_CHAMP
                            FROM `{project_id}.WORK.STRUCTURE_TABLE`
                            WHERE TYP_STRUCTURE_FIC = '{NOM_COURT}'
                                AND FLG_EXCLUSION_DIFF = 'N'
                            ORDER BY POSITION_CHAMP
                        """
                        df_all_cle_comp = client.query(query_all_cle_comp).to_dataframe()
                        ALL_CLE_COMP_t = ', '.join(df_all_cle_comp['NOM_CHAMP'].tolist())
                    except Exception as e:
                        logger.error(f"Error getting comparable columns: {str(e)}")
                        raise
                    
                    # Get key columns with different formats
                    try:
                        query_cle_formats = f"""
                            SELECT NOM_CHAMP
                            FROM `{project_id}.WORK.STRUCTURE_TABLE`
                            WHERE TYP_STRUCTURE_FIC = '{NOM_COURT}'
                                AND FLG_CLE = 'O'
                            ORDER BY POSITION_CHAMP
                        """
                        df_cle_formats = client.query(query_cle_formats).to_dataframe()
                        
                        ALL_CLE = ' '.join(df_cle_formats['NOM_CHAMP'].tolist())
                        ALL_CLE_VIRG = ', '.join(df_cle_formats['NOM_CHAMP'].tolist())
                        ALL_CLE_HASH = ', '.join([f'"{col}"' for col in df_cle_formats['NOM_CHAMP'].tolist()])
                    except Exception as e:
                        logger.error(f"Error getting key column formats: {str(e)}")
                        raise
                    
                    logger.info(f"Comparable columns: {ALL_CLE_COMP_t}")
                    
                    # Find differences in old keys (records that changed)
                    try:
                        query_diff = f"""
                            CREATE OR REPLACE TABLE `{project_id}.WORK.DIFF_OLD_CLE_TO_INSERT{i}` AS
                            SELECT {ALL_CLE_COMP_t}
                            FROM `{project_id}.WORK._ANCIENNES_CLES{i}`
                            EXCEPT DISTINCT
                            SELECT {ALL_CLE_COMP_t}
                            FROM `{project_id}.WORK._ALL_DONNEES_{i}`
                            ORDER BY {ALL_CLE_VIRG}
                        """
                        client.query(query_diff)
                    except Exception as e:
                        logger.error(f"Error finding differences: {str(e)}")
                        raise
                    
                    # Sort anciennes_cles for merge
                    try:
                        query_sort_anciennes = f"""
                            CREATE OR REPLACE TABLE `{project_id}.WORK._ANCIENNES_CLES{i}_SORTED` AS
                            SELECT *
                            FROM `{project_id}.WORK._ANCIENNES_CLES{i}`
                            ORDER BY {ALL_CLE_VIRG}
                        """
                        client.query(query_sort_anciennes)
                    except Exception as e:
                        logger.error(f"Error sorting anciennes_cles: {str(e)}")
                        raise
                    
                    # Identify records to insert (changed) and records to update DAT_DERN_RECPT (unchanged)
                    try:
                        # Load data for merge
                        query_load_anciennes = f"""
                            SELECT *
                            FROM `{project_id}.WORK._ANCIENNES_CLES{i}_SORTED`
                        """
                        df_anciennes_sorted = client.query(query_load_anciennes).to_dataframe()
                        
                        query_load_diff = f"""
                            SELECT {ALL_CLE_VIRG}
                            FROM `{project_id}.WORK.DIFF_OLD_CLE_TO_INSERT{i}`
                        """
                        df_diff = client.query(query_load_diff).to_dataframe()
                        
                        # Merge to identify changed vs unchanged
                        df_merge_diff = df_anciennes_sorted.merge(
                            df_diff,
                            on=key_columns,
                            how='left',
                            indicator=True
                        )
                        
                        # Changed records (to insert with new version)
                        df_old_cle_to_insert = df_merge_diff[df_merge_diff['_merge'] == 'both'].copy()
                        
                        # Unchanged records (to update DAT_DERN_RECPT only)
                        df_maj_date_recp = df_merge_diff[df_merge_diff['_merge'] == 'left_only'][key_columns].copy()
                        
                        # Upload results
                        client.upload_df_to_bq(
                            df=df_old_cle_to_insert.drop(columns=['_merge']),
                            dataset="WORK",
                            table_name=f"_OLD_CLE_TO_INSERT{i}",
                            write_disposition='WRITE_TRUNCATE'
                        )
                        
                        client.upload_df_to_bq(
                            df=df_maj_date_recp,
                            dataset="WORK",
                            table_name=f"_MAJ_DATE_RECP{i}",
                            write_disposition='WRITE_TRUNCATE'
                        )
                    except Exception as e:
                        logger.error(f"Error identifying changed/unchanged records: {str(e)}")
                        raise
                    
                    # Prepare old keys to insert with versioning columns
                    try:
                        query_load_old_insert = f"""
                            SELECT *
                            FROM `{project_id}.WORK._OLD_CLE_TO_INSERT{i}`
                        """
                        df_old_insert = client.query(query_load_old_insert).to_dataframe()
                        
                        if len(df_old_insert) > 0:
                            df_old_insert['DAT_TRT'] = today_dttm
                            df_old_insert['DAT_DEB_VAL'] = DAT_DEB_VAL
                            df_old_insert['DAT_FIN_VAL'] = pd.to_datetime('9999-12-31')
                            df_old_insert['COD_ETA'] = 1
                            df_old_insert['DAT_DERN_RECPT'] = DAT_DEB_VAL
                            
                            # Main table
                            cols_main = ['DAT_TRT', 'DAT_DEB_VAL', 'DAT_FIN_VAL', 'COD_ETA', 'NUM_SEQ', 'DAT_DERN_RECPT']
                            for field_name, info in field_info.items():
                                if info['flg_cle'] == 'O' or info['flg_conf'] == 'N' or info['flg_all_lib'] == 'O':
                                    cols_main.append(field_name)
                            
                            df_old_insert_main = df_old_insert[cols_main]
                            client.upload_df_to_bq(
                                df=df_old_insert_main,
                                dataset="WORK",
                                table_name=f"_OLD_CLE_TO_INSERT{i}",
                                write_disposition='WRITE_TRUNCATE'
                            )
                            
                            # Confidential tables
                            for conf_level in range(1, 5):
                                cols_conf = ['DAT_TRT', 'DAT_DEB_VAL', 'DAT_FIN_VAL', 'COD_ETA', 'NUM_SEQ', 'DAT_DERN_RECPT']
                                for field_name, info in field_info.items():
                                    if (info['flg_cle'] == 'O' or 
                                        info['flg_all_lib'] == 'O' or 
                                        (info['flg_conf'] == 'O' and info['niv_conf'] == conf_level)):
                                        cols_conf.append(field_name)
                                
                                df_old_insert_conf = df_old_insert[cols_conf]
                                client.upload_df_to_bq(
                                    df=df_old_insert_conf,
                                    dataset="WORK",
                                    table_name=f"_OLD_CLE_TO_INSERT_CONF{i}_{conf_level}",
                                    write_disposition='WRITE_TRUNCATE'
                                )
                    except Exception as e:
                        logger.error(f"Error preparing old keys to insert: {str(e)}")
                        raise
                    
                    # Update DAT_DERN_RECPT for unchanged records in main ODS table
                    try:
                        # Build key matching condition
                        key_match_conditions = []
                        for key_col in key_columns:
                            key_match_conditions.append(f't1.{key_col} = t2.{key_col}')
                        key_match_str = ' AND '.join(key_match_conditions)
                        
                        query_update_date_recp = f"""
                            UPDATE `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}` t1
                            SET DAT_DERN_RECPT = DATE('{DAT_DEB_VAL}')
                            WHERE EXISTS (
                                SELECT 1
                                FROM `{project_id}.WORK._MAJ_DATE_RECP{i}` t2
                                WHERE {key_match_str}
                            )
                            AND t1.COD_ETA = 1
                        """
                        client.query(query_update_date_recp)
                    except Exception as e:
                        logger.error(f"Error updating DAT_DERN_RECPT in main table: {str(e)}")
                        raise
                    
                    # Append new keys to main ODS table
                    try:
                        query_append_new = f"""
                            INSERT INTO `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}`
                            SELECT *
                            FROM `{project_id}.WORK._NOUVELLES_CLES{i}`
                        """
                        client.query(query_append_new)
                    except Exception as e:
                        logger.error(f"Error appending new keys to main table: {str(e)}")
                        raise
                    
                    # Append old keys (changed records) to main ODS table
                    try:
                        query_append_old = f"""
                            INSERT INTO `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}`
                            SELECT *
                            FROM `{project_id}.WORK._OLD_CLE_TO_INSERT{i}`
                        """
                        client.query(query_append_old)
                    except Exception as e:
                        logger.error(f"Error appending changed keys to main table: {str(e)}")
                        raise
                    
                    # Process confidential tables
                    if NOM_LIB_ODS_C and NOM_LIB_ODS_C.strip():
                        for conf_level in range(1, 5):
                            conf_table_exists = client.table_exists(f"{NOM_LIB_ODS_C}.{conf_level}", NOM_TAB_ODS)
                            
                            if conf_table_exists:
                                # Update DAT_DERN_RECPT for unchanged records
                                try:
                                    query_update_conf_date = f"""
                                        UPDATE `{project_id}.{NOM_LIB_ODS_C}.{conf_level}.{NOM_TAB_ODS}` t1
                                        SET DAT_DERN_RECPT = DATE('{DAT_DEB_VAL}')
                                        WHERE EXISTS (
                                            SELECT 1
                                            FROM `{project_id}.WORK._MAJ_DATE_RECP{i}` t2
                                            WHERE {key_match_str}
                                        )
                                        AND t1.COD_ETA = 1
                                    """
                                    client.query(query_update_conf_date)
                                except Exception as e:
                                    logger.warning(f"Error updating DAT_DERN_RECPT in conf table level {conf_level}: {str(e)}")
                                
                                # Append new keys
                                try:
                                    query_append_conf_new = f"""
                                        INSERT INTO `{project_id}.{NOM_LIB_ODS_C}.{conf_level}.{NOM_TAB_ODS}`
                                        SELECT *
                                        FROM `{project_id}.WORK._NOUVELLES_CLES_CONF{i}_{conf_level}`
                                    """
                                    client.query(query_append_conf_new)
                                except Exception as e:
                                    logger.warning(f"Error appending new keys to conf table level {conf_level}: {str(e)}")
                                
                                # Append changed keys
                                try:
                                    query_append_conf_old = f"""
                                        INSERT INTO `{project_id}.{NOM_LIB_ODS_C}.{conf_level}.{NOM_TAB_ODS}`
                                        SELECT *
                                        FROM `{project_id}.WORK._OLD_CLE_TO_INSERT_CONF{i}_{conf_level}`
                                    """
                                    client.query(query_append_conf_old)
                                except Exception as e:
                                    logger.warning(f"Error appending changed keys to conf table level {conf_level}: {str(e)}")
                    
                    # Find last key column for logging
                    DERN_CLE = None
                    for field_name, info in field_info.items():
                        if info['flg_cle'] == 'O':
                            DERN_CLE = field_name
                    
                    logger.info(f"Last key column: {DERN_CLE}")
                    
                    # Sort final ODS table by keys and descending DAT_DEB_VAL
                    try:
                        query_final_sort = f"""
                            CREATE OR REPLACE TABLE `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}_TEMP` AS
                            SELECT *
                            FROM `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}`
                            ORDER BY {ALL_CLE_VIRG}, DAT_DEB_VAL DESC
                        """
                        client.query(query_final_sort)
                        
                        # Replace original table
                        query_replace = f"""
                            CREATE OR REPLACE TABLE `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}` AS
                            SELECT *
                            FROM `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}_TEMP`
                        """
                        client.query(query_replace)
                        
                        # Delete temp table
                        client.delete_table(NOM_LIB_ODS, f"{NOM_TAB_ODS}_TEMP", project_id)
                    except Exception as e:
                        logger.error(f"Error sorting final ODS table: {str(e)}")
                        raise
# Update DAT_FIN_VAL using LAG function to close out previous versions
                    # Sort by keys descending DAT_DEB_VAL, then use LAG to get next DAT_DEB_VAL
                    try:
                        query_update_dat_fin = f"""
                            CREATE OR REPLACE TABLE `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}` AS
                            WITH sorted_data AS (
                                SELECT *,
                                    LAG(DAT_DEB_VAL) OVER (
                                        PARTITION BY {ALL_CLE_VIRG}
                                        ORDER BY DAT_DEB_VAL DESC
                                    ) AS DAT_FIN_VAL2
                                FROM `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}`
                            ),
                            updated_data AS (
                                SELECT
                                    * EXCEPT(DAT_FIN_VAL2, DAT_FIN_VAL),
                                    CASE
                                        WHEN DAT_FIN_VAL2 IS NOT NULL AND DAT_FIN_VAL = DATE('9999-12-31')
                                        THEN DATE_SUB(DAT_FIN_VAL2, INTERVAL 1 DAY)
                                        ELSE DAT_FIN_VAL
                                    END AS DAT_FIN_VAL
                                FROM sorted_data
                            )
                            SELECT * EXCEPT(DAT_FIN_VAL2)
                            FROM updated_data
                            ORDER BY {ALL_CLE_VIRG}, DAT_DEB_VAL DESC
                        """
                        client.query(query_update_dat_fin)
                    except Exception as e:
                        logger.error(f"Error updating DAT_FIN_VAL for {NOM_TAB_ODS}: {str(e)}")
                        raise
                    
                    # Sort by keys and DAT_DEB_VAL ascending
                    try:
                        query_sort_asc = f"""
                            CREATE OR REPLACE TABLE `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}_TEMP` AS
                            SELECT *
                            FROM `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}`
                            ORDER BY {ALL_CLE_VIRG}, DAT_DEB_VAL
                        """
                        client.query(query_sort_asc)
                        
                        query_replace_sorted = f"""
                            CREATE OR REPLACE TABLE `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}` AS
                            SELECT *
                            FROM `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}_TEMP`
                        """
                        client.query(query_replace_sorted)
                        
                        client.delete_table(NOM_LIB_ODS, f"{NOM_TAB_ODS}_TEMP", project_id)
                    except Exception as e:
                        logger.error(f"Error sorting table ascending: {str(e)}")
                        raise
                    
                    # Recalculate NUM_SEQ and COD_ETA
                    try:
                        query_update_seq = f"""
                            CREATE OR REPLACE TABLE `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}` AS
                            WITH numbered_data AS (
                                SELECT *,
                                    ROW_NUMBER() OVER (
                                        PARTITION BY {ALL_CLE_VIRG}
                                        ORDER BY DAT_DEB_VAL
                                    ) AS NUM_SEQ2,
                                    ROW_NUMBER() OVER (
                                        PARTITION BY {ALL_CLE_VIRG}
                                        ORDER BY DAT_DEB_VAL DESC
                                    ) AS IS_LAST
                                FROM `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}`
                            )
                            SELECT
                                * EXCEPT(NUM_SEQ, NUM_SEQ2, COD_ETA, IS_LAST),
                                NUM_SEQ2 AS NUM_SEQ,
                                CASE WHEN IS_LAST = 1 THEN 1 ELSE 0 END AS COD_ETA
                            FROM numbered_data
                        """
                        client.query(query_update_seq)
                    except Exception as e:
                        logger.error(f"Error updating NUM_SEQ and COD_ETA: {str(e)}")
                        raise
                    
                    # Process confidential tables with same logic
                    if NOM_LIB_ODS_C and NOM_LIB_ODS_C.strip():
                        for conf_level in range(1, 5):
                            conf_table_exists = client.table_exists(f"{NOM_LIB_ODS_C}.{conf_level}", NOM_TAB_ODS)
                            
                            if conf_table_exists:
                                # Sort descending
                                try:
                                    query_sort_conf_desc = f"""
                                        CREATE OR REPLACE TABLE `{project_id}.{NOM_LIB_ODS_C}.{conf_level}.{NOM_TAB_ODS}_TEMP` AS
                                        SELECT *
                                        FROM `{project_id}.{NOM_LIB_ODS_C}.{conf_level}.{NOM_TAB_ODS}`
                                        ORDER BY {ALL_CLE_VIRG}, DAT_DEB_VAL DESC
                                    """
                                    client.query(query_sort_conf_desc)
                                    
                                    query_replace_conf = f"""
                                        CREATE OR REPLACE TABLE `{project_id}.{NOM_LIB_ODS_C}.{conf_level}.{NOM_TAB_ODS}` AS
                                        SELECT *
                                        FROM `{project_id}.{NOM_LIB_ODS_C}.{conf_level}.{NOM_TAB_ODS}_TEMP`
                                    """
                                    client.query(query_replace_conf)
                                    
                                    client.delete_table(f"{NOM_LIB_ODS_C}.{conf_level}", f"{NOM_TAB_ODS}_TEMP", project_id)
                                except Exception as e:
                                    logger.warning(f"Error sorting conf table level {conf_level} descending: {str(e)}")
                                
                                # Update DAT_FIN_VAL
                                try:
                                    query_update_conf_fin = f"""
                                        CREATE OR REPLACE TABLE `{project_id}.{NOM_LIB_ODS_C}.{conf_level}.{NOM_TAB_ODS}` AS
                                        WITH sorted_data AS (
                                            SELECT *,
                                                LAG(DAT_DEB_VAL) OVER (
                                                    PARTITION BY {ALL_CLE_VIRG}
                                                    ORDER BY DAT_DEB_VAL DESC
                                                ) AS DAT_FIN_VAL2
                                            FROM `{project_id}.{NOM_LIB_ODS_C}.{conf_level}.{NOM_TAB_ODS}`
                                        )
                                        SELECT
                                            * EXCEPT(DAT_FIN_VAL2, DAT_FIN_VAL),
                                            CASE
                                                WHEN DAT_FIN_VAL2 IS NOT NULL AND DAT_FIN_VAL = DATE('9999-12-31')
                                                THEN DATE_SUB(DAT_FIN_VAL2, INTERVAL 1 DAY)
                                                ELSE DAT_FIN_VAL
                                            END AS DAT_FIN_VAL
                                        FROM sorted_data
                                    """
                                    client.query(query_update_conf_fin)
                                except Exception as e:
                                    logger.warning(f"Error updating DAT_FIN_VAL for conf level {conf_level}: {str(e)}")
                                
                                # Sort ascending
                                try:
                                    query_sort_conf_asc = f"""
                                        CREATE OR REPLACE TABLE `{project_id}.{NOM_LIB_ODS_C}.{conf_level}.{NOM_TAB_ODS}_TEMP` AS
                                        SELECT *
                                        FROM `{project_id}.{NOM_LIB_ODS_C}.{conf_level}.{NOM_TAB_ODS}`
                                        ORDER BY {ALL_CLE_VIRG}, DAT_DEB_VAL
                                    """
                                    client.query(query_sort_conf_asc)
                                    
                                    query_replace_conf_asc = f"""
                                        CREATE OR REPLACE TABLE `{project_id}.{NOM_LIB_ODS_C}.{conf_level}.{NOM_TAB_ODS}` AS
                                        SELECT *
                                        FROM `{project_id}.{NOM_LIB_ODS_C}.{conf_level}.{NOM_TAB_ODS}_TEMP`
                                    """
                                    client.query(query_replace_conf_asc)
                                    
                                    client.delete_table(f"{NOM_LIB_ODS_C}.{conf_level}", f"{NOM_TAB_ODS}_TEMP", project_id)
                                except Exception as e:
                                    logger.warning(f"Error sorting conf level {conf_level} ascending: {str(e)}")
                                
                                # Update NUM_SEQ and COD_ETA
                                try:
                                    query_update_conf_seq = f"""
                                        CREATE OR REPLACE TABLE `{project_id}.{NOM_LIB_ODS_C}.{conf_level}.{NOM_TAB_ODS}` AS
                                        WITH numbered_data AS (
                                            SELECT *,
                                                ROW_NUMBER() OVER (
                                                    PARTITION BY {ALL_CLE_VIRG}
                                                    ORDER BY DAT_DEB_VAL
                                                ) AS NUM_SEQ2,
                                                ROW_NUMBER() OVER (
                                                    PARTITION BY {ALL_CLE_VIRG}
                                                    ORDER BY DAT_DEB_VAL DESC
                                                ) AS IS_LAST
                                            FROM `{project_id}.{NOM_LIB_ODS_C}.{conf_level}.{NOM_TAB_ODS}`
                                        )
                                        SELECT
                                            * EXCEPT(NUM_SEQ, NUM_SEQ2, COD_ETA, IS_LAST),
                                            NUM_SEQ2 AS NUM_SEQ,
                                            CASE WHEN IS_LAST = 1 THEN 1 ELSE 0 END AS COD_ETA
                                        FROM numbered_data
                                    """
                                    client.query(query_update_conf_seq)
                                except Exception as e:
                                    logger.warning(f"Error updating NUM_SEQ/COD_ETA for conf level {conf_level}: {str(e)}")
                    
                    # Get NOM_COURT for column list building
                    try:
                        query_nom_court = f"""
                            SELECT TYP_STRUCTURE_FIC
                            FROM `{project_id}.S_DACPRM.ACO_TAB_ODS`
                            WHERE NOM_TAB_TAMPON = '{NOM_TAB_TAMPON}'
                            LIMIT 1
                        """
                        df_nom_court = client.query(query_nom_court).to_dataframe()
                        NOM_COURT = df_nom_court.iloc[0]['TYP_STRUCTURE_FIC'] if not df_nom_court.empty else TYP_TAB_ODS
                    except Exception as e:
                        logger.warning(f"Error getting NOM_COURT: {str(e)}")
                        NOM_COURT = TYP_TAB_ODS
                    
                    # Build column lists
                    try:
                        # All columns
                        query_all_cols = f"""
                            SELECT NOM_CHAMP
                            FROM `{project_id}.WORK.STRUCTURE_TABLE`
                            WHERE TYP_STRUCTURE_FIC = '{NOM_COURT}'
                            ORDER BY POSITION_CHAMP
                        """
                        df_all_cols = client.query(query_all_cols).to_dataframe()
                        ALL_CLE_COMP = ', '.join(df_all_cols['NOM_CHAMP'].tolist())
                        
                        # Columns to exclude from comparison
                        query_excl_cols = f"""
                            SELECT NOM_CHAMP
                            FROM `{project_id}.WORK.STRUCTURE_TABLE`
                            WHERE TYP_STRUCTURE_FIC = '{NOM_COURT}'
                                AND FLG_EXCLUSION_DIFF = 'O'
                            ORDER BY POSITION_CHAMP
                        """
                        df_excl_cols = client.query(query_excl_cols).to_dataframe()
                        ALL_CLE_AMAJ = ', '.join(df_excl_cols['NOM_CHAMP'].tolist())
                        
                        # Key columns
                        query_key_cols = f"""
                            SELECT NOM_CHAMP
                            FROM `{project_id}.WORK.STRUCTURE_TABLE`
                            WHERE TYP_STRUCTURE_FIC = '{NOM_COURT}'
                                AND FLG_CLE = 'O'
                            ORDER BY POSITION_CHAMP
                        """
                        df_key_cols = client.query(query_key_cols).to_dataframe()
                        ALL_CLE = ', '.join(df_key_cols['NOM_CHAMP'].tolist())
                        ALL_CLE_s = ' '.join(df_key_cols['NOM_CHAMP'].tolist())
                    except Exception as e:
                        logger.error(f"Error building column lists: {str(e)}")
                        raise
                    
                    logger.info(f"Exclusion columns length: {len(ALL_CLE_AMAJ)}")
                    
                    # Count new and old keys
                    try:
                        query_count_new_keys = f"""
                            SELECT COUNT(*) as NB_NEW_CLE
                            FROM `{project_id}.WORK._NOUVELLES_CLES{i}`
                        """
                        df_count_new_keys = client.query(query_count_new_keys).to_dataframe()
                        NB_NEW_CLE = int(df_count_new_keys.iloc[0]['NB_NEW_CLE'])
                    except Exception as e:
                        logger.error(f"Error counting new keys: {str(e)}")
                        raise
                    
                    try:
                        query_count_old_keys = f"""
                            SELECT COUNT(*) as NB_OLD_CLE
                            FROM `{project_id}.WORK._OLD_CLE_TO_INSERT{i}`
                        """
                        df_count_old_keys = client.query(query_count_old_keys).to_dataframe()
                        NB_OLD_CLE = int(df_count_old_keys.iloc[0]['NB_OLD_CLE'])
                    except Exception as e:
                        logger.error(f"Error counting old keys: {str(e)}")
                        raise
                    
                    # Log results
                    today_dttm = dt.datetime.now()
                    try:
                        util_dec_aco_alim_suivi_trt(
                            ID_TRT=MV_ID_TRT,
                            DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                            NOM_TRT="CHARGEMENT_ODS",
                            DAT_REEL_TRT=today_dttm,
                            MESG=f"La table {NOM_TAB_TAMPON} est chargé dans ODS : {NB_NEW_CLE} NOUVELLE(S) CLE(S) et {NB_OLD_CLE} MODIFICATION(S) CLES EXISTANTES.",
                            CD_RETOUR=0
                        )
                    except Exception as e:
                        logger.error(f"Error logging results: {str(e)}")
                    
                    # Get MV_EXTENSION from params
                    MV_EXTENSION = params.get("MV_EXTENSION", "")
                    
                    # Move file to archives
                    try:
                        source_path = f"{REF_FIC_ENT}/ok/{NOM_FIC}"
                        dest_path = f"{REF_FIC_ENT}/archives/{NOM_FIC}.{MV_EXTENSION}"
                        
                        if source_path.startswith("gs://"):
                            from google.cloud import storage
                            
                            source_parts = source_path.replace("gs://", "").split("/", 1)
                            dest_parts = dest_path.replace("gs://", "").split("/", 1)
                            
                            source_bucket = source_parts[0]
                            source_blob = source_parts[1]
                            dest_bucket = dest_parts[0]
                            dest_blob = dest_parts[1]
                            
                            storage_client_gcs = storage.Client(project=project_id)
                            source_bucket_obj = storage_client_gcs.bucket(source_bucket)
                            source_blob_obj = source_bucket_obj.blob(source_blob)
                            dest_bucket_obj = storage_client_gcs.bucket(dest_bucket)
                            
                            source_bucket_obj.copy_blob(source_blob_obj, dest_bucket_obj, dest_blob)
                            source_blob_obj.delete()
                            
                            move_success = True
                        else:
                            import shutil
                            shutil.move(source_path, dest_path)
                            move_success = True
                    except Exception as e:
                        logger.warning(f"Error moving file {NOM_FIC}: {str(e)}")
                        move_success = False
                    
                    logger.info(f"***info move_success: {move_success}")
                    
                    if move_success:
                        today_dttm = dt.datetime.now()
                        try:
                            util_dec_aco_alim_suivi_trt(
                                ID_TRT=MV_ID_TRT,
                                DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                                NOM_TRT="CHARGEMENT_ODS",
                                DAT_REEL_TRT=today_dttm,
                                MESG=f"Le fichier {NOM_FIC} a été déplacé dans le répertoire {REF_FIC_ENT}/archives.",
                                CD_RETOUR=0
                            )
                        except Exception as e:
                            logger.error(f"Error logging file move: {str(e)}")
                        
                        # Zip the file
                        try:
                            archive_file = f"{NOM_FIC}.{MV_EXTENSION}"
                            archive_path = f"{REF_FIC_ENT}/archives/{archive_file}"
                            zip_path = f"{REF_FIC_ENT}/archives/Archives_{MV_EXTENSION}.zip"
                            
                            if archive_path.startswith("gs://"):
                                import tempfile
                                import zipfile
                                
                                with tempfile.TemporaryDirectory() as tmpdir:
                                    archive_parts = archive_path.replace("gs://", "").split("/", 1)
                                    archive_bucket = archive_parts[0]
                                    archive_blob = archive_parts[1]
                                    
                                    local_file = f"{tmpdir}/{archive_file}"
                                    storage_client_gcs = storage.Client(project=project_id)
                                    bucket_obj = storage_client_gcs.bucket(archive_bucket)
                                    blob_obj = bucket_obj.blob(archive_blob)
                                    blob_obj.download_to_filename(local_file)
                                    
                                    local_zip = f"{tmpdir}/Archives_{MV_EXTENSION}.zip"
                                    
                                    zip_parts = zip_path.replace("gs://", "").split("/", 1)
                                    zip_blob = zip_parts[1]
                                    zip_blob_obj = bucket_obj.blob(zip_blob)
                                    
                                    if zip_blob_obj.exists():
                                        zip_blob_obj.download_to_filename(local_zip)
                                        mode = 'a'
                                    else:
                                        mode = 'w'
                                    
                                    with zipfile.ZipFile(local_zip, mode) as zf:
                                        zf.write(local_file, archive_file)
                                    
                                    zip_blob_obj.upload_from_filename(local_zip)
                                    
                                    zip_success = True
                            else:
                                import os
                                import zipfile
                                
                                mode = 'a' if os.path.exists(zip_path) else 'w'
                                with zipfile.ZipFile(zip_path, mode) as zf:
                                    zf.write(archive_path, archive_file)
                                
                                zip_success = True
                        except Exception as e:
                            logger.warning(f"Error zipping file {NOM_FIC}: {str(e)}")
                            zip_success = False
                        
                        if zip_success:
                            today_dttm = dt.datetime.now()
                            try:
                                util_dec_aco_alim_suivi_trt(
                                    ID_TRT=MV_ID_TRT,
                                    DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                                    NOM_TRT="CHARGEMENT_ODS",
                                    DAT_REEL_TRT=today_dttm,
                                    MESG=f"Le fichier {NOM_FIC} a été ajouté à l'archive Archives_{MV_EXTENSION}.zip.",
                                    CD_RETOUR=0
                                )
                            except Exception as e:
                                logger.error(f"Error logging zip success: {str(e)}")
                            
                            # Remove the unzipped file
                            try:
                                if archive_path.startswith("gs://"):
                                    archive_parts = archive_path.replace("gs://", "").split("/", 1)
                                    archive_bucket = archive_parts[0]
                                    archive_blob = archive_parts[1]
                                    
                                    storage_client_gcs = storage.Client(project=project_id)
                                    bucket_obj = storage_client_gcs.bucket(archive_bucket)
                                    blob_obj = bucket_obj.blob(archive_blob)
                                    blob_obj.delete()
                            except Exception as e:
                                logger.warning(f"Error removing archived file: {str(e)}")
                        else:
                            today_dttm = dt.datetime.now()
                            try:
                                util_dec_aco_alim_suivi_trt(
                                    ID_TRT=MV_ID_TRT,
                                    DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                                    NOM_TRT="CHARGEMENT_ODS",
                                    DAT_REEL_TRT=today_dttm,
                                    MESG=f"Impossible d'ajouter le fichier {NOM_FIC} à l'archive Archives_{MV_EXTENSION}.zip.",
                                    CD_RETOUR=99
                                )
                            except Exception as e:
                                logger.error(f"Error logging zip failure: {str(e)}")
                        
                        # Update tracking table
                        try:
                            query_update_tracking = f"""
                                UPDATE `{project_id}.{lib_dec_aco_suivi}.ACO_SUIVI_CHG_TAMPON_ODS`
                                SET FLG_TRAITE_ODS = 'O',
                                    CD_STATUT_ODS = 'C',
                                    DAT_REEL_TRT_ODS = TIMESTAMP('{today_dttm}')
                                WHERE ID_TRT = {MV_ID_TRT}
                                    AND DAT_BATCH_TRT = {MV_DAT_BATCH_TRT}
                                    AND ID_FIC = '{ID_FIC}'
                            """
                            client.query(query_update_tracking)
                        except Exception as e:
                            logger.error(f"Error updating tracking table: {str(e)}")
                            raise
                    else:
                        today_dttm = dt.datetime.now()
                        try:
                            util_dec_aco_alim_suivi_trt(
                                ID_TRT=MV_ID_TRT,
                                DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                                NOM_TRT="CHARGEMENT_ODS",
                                DAT_REEL_TRT=today_dttm,
                                MESG=f"Le fichier {NOM_FIC} n'a pas pu être déplacé dans le répertoire {REF_FIC_ENT}/archives.",
                                CD_RETOUR=0
                            )
                        except Exception as e:
                            logger.error(f"Error logging move failure: {str(e)}")
            
            # ELSE case: validation errors exist
            else:
                # Count different error types
                try:
                    query_count_err_date = f"""
                        SELECT COUNT(*) as NB_ERR_DATE
                        FROM `{project_id}.WORK._ERR_DATE{i}`
                    """
                    df_count_err_date = client.query(query_count_err_date).to_dataframe()
                    NB_ERR_DATE = int(df_count_err_date.iloc[0]['NB_ERR_DATE'])
                except Exception as e:
                    logger.error(f"Error counting date errors: {str(e)}")
                    NB_ERR_DATE = 0
                
                try:
                    query_count_err_char = f"""
                        SELECT COUNT(*) as NB_ERR_CHAR
                        FROM `{project_id}.WORK._ERR_CHAR{i}`
                    """
                    df_count_err_char = client.query(query_count_err_char).to_dataframe()
                    NB_ERR_CHAR = int(df_count_err_char.iloc[0]['NB_ERR_CHAR'])
                except Exception as e:
                    logger.error(f"Error counting char errors: {str(e)}")
                    NB_ERR_CHAR = 0
                
                try:
                    query_count_err_num = f"""
                        SELECT COUNT(*) as NB_ERR_NUM
                        FROM `{project_id}.WORK._ERR_NUM{i}`
                    """
                    df_count_err_num = client.query(query_count_err_num).to_dataframe()
                    NB_ERR_NUM = int(df_count_err_num.iloc[0]['NB_ERR_NUM'])
                except Exception as e:
                    logger.error(f"Error counting num errors: {str(e)}")
                    NB_ERR_NUM = 0
                
                try:
                    query_count_err_cle = f"""
                        SELECT COUNT(*) as NB_ERR_CLE
                        FROM `{project_id}.WORK._ERR_CLE{i}`
                    """
                    df_count_err_cle = client.query(query_count_err_cle).to_dataframe()
                    NB_ERR_CLE = int(df_count_err_cle.iloc[0]['NB_ERR_CLE'])
                except Exception as e:
                    logger.error(f"Error counting key errors: {str(e)}")
                    NB_ERR_CLE = 0
                
                # Log error summary
                today_dttm = dt.datetime.now()
                try:
                    util_dec_aco_alim_suivi_trt(
                        ID_TRT=MV_ID_TRT,
                        DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                        NOM_TRT="CHARGEMENT_ODS",
                        DAT_REEL_TRT=today_dttm,
                        MESG=f"La table {NOM_TAB_TAMPON} contient des erreurs : {NB_ERR_DATE} Date(s), {NB_ERR_CHAR} Char(s), {NB_ERR_NUM} Num(s), {NB_ERR_CLE} Clé(s) : Pas de chargement.",
                        CD_RETOUR=99
                    )
                except Exception as e:
                    logger.error(f"Error logging error summary: {str(e)}")
                
                # Move file to KO directory
                try:
                    source_path = f"{REF_FIC_ENT}/ok/{NOM_FIC}"
                    dest_path = f"{REF_FIC_KO}/{NOM_FIC}.{MV_EXTENSION}"
                    
                    if source_path.startswith("gs://"):
                        from google.cloud import storage
                        
                        source_parts = source_path.replace("gs://", "").split("/", 1)
                        dest_parts = dest_path.replace("gs://", "").split("/", 1)
                        
                        source_bucket = source_parts[0]
                        source_blob = source_parts[1]
                        dest_bucket = dest_parts[0]
                        dest_blob = dest_parts[1]
                        
                        storage_client_gcs = storage.Client(project=project_id)
                        source_bucket_obj = storage_client_gcs.bucket(source_bucket)
                        source_blob_obj = source_bucket_obj.blob(source_blob)
                        dest_bucket_obj = storage_client_gcs.bucket(dest_bucket)
                        
                        source_bucket_obj.copy_blob(source_blob_obj, dest_bucket_obj, dest_blob)
                        source_blob_obj.delete()
                        
                        move_ko_success = True
                    else:
                        import shutil
                        shutil.move(source_path, dest_path)
                        move_ko_success = True
                except Exception as e:
                    logger.warning(f"Error moving file {NOM_FIC} to KO: {str(e)}")
                    move_ko_success = False
                
                today_dttm = dt.datetime.now()
                
                if move_ko_success:
                    try:
                        util_dec_aco_alim_suivi_trt(
                            ID_TRT=MV_ID_TRT,
                            DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                            NOM_TRT="CHARGEMENT_ODS",
                            DAT_REEL_TRT=today_dttm,
                            MESG=f"Le fichier {NOM_FIC} a été déplacé dans le répertoire {REF_FIC_KO}.",
                            CD_RETOUR=99
                        )
                    except Exception as e:
                        logger.error(f"Error logging KO move success: {str(e)}")
                else:
                    try:
                        util_dec_aco_alim_suivi_trt(
                            ID_TRT=MV_ID_TRT,
                            DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                            NOM_TRT="CHARGEMENT_ODS",
                            DAT_REEL_TRT=today_dttm,
                            MESG=f"Le fichier {NOM_FIC} n'a pas pu être déplacé dans le répertoire {REF_FIC_KO}.",
                            CD_RETOUR=99
                        )
                    except Exception as e:
                        logger.error(f"Error logging KO move failure: {str(e)}")
                
                # Update tracking table with error status
                try:
                    query_update_error = f"""
                        UPDATE `{project_id}.{lib_dec_aco_suivi}.ACO_SUIVI_CHG_TAMPON_ODS`
                        SET CD_STATUT_ODS = 'E',
                            DAT_REEL_TRT_ODS = TIMESTAMP('{today_dttm}')
                        WHERE ID_TRT = {MV_ID_TRT}
                            AND DAT_BATCH_TRT = {MV_DAT_BATCH_TRT}
                            AND ID_FIC = '{ID_FIC}'
                    """
                    client.query(query_update_error)
                except Exception as e:
                    logger.error(f"Error updating tracking table with error status: {str(e)}")
                    raise
        
        # ELSE case: table has no keys but is PDK_SIEG system
    elif NB_CLE == 0 and ID_SYS_GEST == "PDK_SIEG":
        # Remove duplicates based on ID_LIGNE and DT_ARRETE concatenation
        try:
            query_dedupe_pdk = f"""
                CREATE OR REPLACE TABLE `{project_id}.WORK.{NOM_TAB_TAMPON}` AS
                SELECT *
                FROM `{project_id}.{NOM_LIB_TAMPON}.{NOM_TAB_TAMPON}` t1
                GROUP BY CONCAT(CAST(ID_LIGNE AS STRING), '-', CAST(DT_ARRETE AS STRING))
                HAVING COUNT(CONCAT(CAST(ID_LIGNE AS STRING), '-', CAST(DT_ARRETE AS STRING))) = 1
            """
            client.query(query_dedupe_pdk)
        except Exception as e:
            logger.error(f"Error deduplicating PDK_SIEG table {NOM_TAB_TAMPON}: {str(e)}")
            raise
        
        # Identify duplicates
        try:
            query_duplicates_pdk = f"""
                CREATE OR REPLACE TABLE `{project_id}.WORK.DBL_{i}` AS
                SELECT DISTINCT *
                FROM `{project_id}.{NOM_LIB_TAMPON}.{NOM_TAB_TAMPON}` t1
                GROUP BY CONCAT(CAST(ID_LIGNE AS STRING), '-', CAST(DT_ARRETE AS STRING))
                HAVING COUNT(CONCAT(CAST(ID_LIGNE AS STRING), '-', CAST(DT_ARRETE AS STRING))) > 1
            """
            client.query(query_duplicates_pdk)
        except Exception as e:
            logger.error(f"Error identifying duplicates in PDK_SIEG table: {str(e)}")
            raise
        
        # Count duplicates
        try:
            query_count_dupl_pdk = f"""
                SELECT COUNT(*) as NB_DOUBL
                FROM `{project_id}.WORK.DBL_{i}`
            """
            df_count_dupl_pdk = client.query(query_count_dupl_pdk).to_dataframe()
            NB_DOUBL = int(df_count_dupl_pdk.iloc[0]['NB_DOUBL'])
        except Exception as e:
            logger.error(f"Error counting PDK_SIEG duplicates: {str(e)}")
            raise
        
        if NB_DOUBL > 0:
            FLG_DBL = "O"
            params["FLG_DBL"] = FLG_DBL
        
        # Log duplicates if any
        if NB_DOUBL > 0:
            today_dttm = dt.datetime.now()
            try:
                query_insert_dupl_pdk = f"""
                    INSERT INTO `{project_id}.{lib_dec_aco_suivi}.ACO_SUIVI_DOUBLONS`
                    (ID_TRT, DAT_BATCH_TRT, DAT_REEL_TRT, TABLE, CLE, NB_DOUBLONS, NB_OBS_TABLE)
                    VALUES ({MV_ID_TRT}, '{MV_DAT_BATCH_TRT_DATE}', TIMESTAMP('{today_dttm}'),
                            '{NOM_TAB_TAMPON}', 'ID_LIGNE-DT_ARRETE', {NB_DOUBL}, {NB_OBS})
                """
                client.query(query_insert_dupl_pdk)
            except Exception as e:
                logger.error(f"Error logging PDK_SIEG duplicates: {str(e)}")
                raise
            
            try:
                util_dec_aco_alim_suivi_trt(
                    ID_TRT=MV_ID_TRT,
                    DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                    NOM_TRT="CHARGEMENT_ODS",
                    DAT_REEL_TRT=today_dttm,
                    MESG=f"La table {NOM_TAB_TAMPON} contient {NB_DOUBL} doublons",
                    CD_RETOUR=0
                )
            except Exception as e:
                logger.error(f"Error logging PDK_SIEG duplicate message: {str(e)}")
# Process data validation for PDK_SIEG tables
        try:
            # Load staging table to pandas for complex processing
            query_load_tampon_pdk = f"""
                SELECT *
                FROM `{project_id}.WORK.{NOM_TAB_TAMPON}`
            """
            df_tampon_pdk = client.query(query_load_tampon_pdk).to_dataframe()
        except Exception as e:
            logger.error(f"Error loading PDK_SIEG staging table to pandas: {str(e)}")
            raise
        
        # Process data validation and conversion
        df_ods_pdk, df_erreur_pdk, df_err_char_pdk, df_err_date_pdk, df_err_num_pdk, df_err_cle_pdk = process_table_validation(
            df_tampon_pdk, field_info, NB_VAR_C, NB_VAR_D, NB_VAR_N, NB_CLE_C, NB_CLE_N
        )
        
        # Add ID_LIGNE and DT_ARRETE columns
        df_ods_pdk['ID_LIGNE'] = range(1, len(df_ods_pdk) + 1)
        
        # Calculate DT_ARRETE as last day of previous month
        try:

            from dateutil.relativedelta import relativedelta
            
            batch_date = dt.datetime.strptime(str(MV_DAT_BATCH_TRT), '%Y%m%d')
            # Get last day of previous month
            first_day_current_month = batch_date.replace(day=1)
            last_day_prev_month = first_day_current_month - relativedelta(days=1)
            df_ods_pdk['DT_ARRETE'] = last_day_prev_month
        except Exception as e:
            logger.error(f"Error calculating DT_ARRETE: {str(e)}")
            raise
        
        # Filter errors for PDK tables only
        # Only output errors if table name starts with "PDK_"
        if NOM_TAB_TAMPON.startswith("PDK_"):
            NB_ERR_PDK = len(df_erreur_pdk)
        else:
            NB_ERR_PDK = 0
            df_erreur_pdk = pd.DataFrame()
            df_err_char_pdk = pd.DataFrame()
            df_err_date_pdk = pd.DataFrame()
            df_err_num_pdk = pd.DataFrame()
            df_err_cle_pdk = pd.DataFrame()
        
        # Process only if no errors
        if NB_ERR_PDK == 0:
            logger.info(f"PDK_SIEG table {NOM_TAB_TAMPON} has no errors, proceeding to load ODS")
            
            # Delete temporary staging table
            try:
                client.delete_table("WORK", NOM_TAB_TAMPON, project_id)
            except Exception as e:
                logger.warning(f"Error deleting temporary PDK table {NOM_TAB_TAMPON}: {str(e)}")
            
            # Check if ODS table exists
            table_exists = client.table_exists(NOM_LIB_ODS, NOM_TAB_ODS)
            
            if not table_exists:
                logger.info(f"Creating new PDK_SIEG ODS table: {NOM_LIB_ODS}.{NOM_TAB_ODS}")
                
                # Add versioning columns
                today_dttm = dt.datetime.now()
                
                # Determine DAT_DEB_VAL based on system
                if ID_SYS_GEST == "GFP":
                    DAT_DEB_VAL = params.get(f"MV_DAT_ARRETE_{ID_SYS_GEST}")
                else:
                    DAT_DEB_VAL = MV_DAT_BATCH_TRT
                
                df_ods_pdk['DAT_TRT'] = today_dttm
                df_ods_pdk['DAT_DEB_VAL'] = DAT_DEB_VAL
                df_ods_pdk['DAT_FIN_VAL'] = pd.to_datetime('9999-12-31')
                df_ods_pdk['COD_ETA'] = 1
                df_ods_pdk['NUM_SEQ'] = 1
                df_ods_pdk['DAT_DERN_RECPT'] = DAT_DEB_VAL
                
                # Select columns for main ODS table (non-confidential or key columns)
                cols_main_pdk = ['ID_LIGNE', 'DT_ARRETE', 'DAT_TRT', 'DAT_DEB_VAL', 'DAT_FIN_VAL', 'COD_ETA', 'NUM_SEQ', 'DAT_DERN_RECPT']
                for field_name, info in field_info.items():
                    if info['flg_cle'] == 'O' or info['flg_conf'] == 'N' or info['flg_all_lib'] == 'O':
                        cols_main_pdk.append(field_name)
                
                df_ods_main_pdk = df_ods_pdk[cols_main_pdk]
                
                # Upload main ODS table
                try:
                    client.upload_df_to_bq(
                        df=df_ods_main_pdk,
                        dataset=NOM_LIB_ODS,
                        table_name=NOM_TAB_ODS,
                        write_disposition='WRITE_TRUNCATE'
                    )
                except Exception as e:
                    logger.error(f"Error uploading main PDK_SIEG ODS table: {str(e)}")
                    raise
                
                # Create confidential tables if needed
                if NOM_LIB_ODS_C and NOM_LIB_ODS_C.strip():
                    SIZE_CLE = 2  # ID_LIGNE and DT_ARRETE
                    SIZE_MIN_BASE = SIZE_CLE + 6
                    
                    for conf_level in range(1, 5):
                        # Select columns for this confidentiality level
                        cols_conf_pdk = ['ID_LIGNE', 'DT_ARRETE', 'DAT_TRT', 'DAT_DEB_VAL', 'DAT_FIN_VAL', 'COD_ETA', 'NUM_SEQ', 'DAT_DERN_RECPT']
                        for field_name, info in field_info.items():
                            if (info['flg_cle'] == 'O' or 
                                info['flg_all_lib'] == 'O' or 
                                (info['flg_conf'] == 'O' and info['niv_conf'] == conf_level)):
                                cols_conf_pdk.append(field_name)
                        
                        # Only create if there are confidential columns
                        if len(cols_conf_pdk) > SIZE_MIN_BASE:
                            df_ods_conf_pdk = df_ods_pdk[cols_conf_pdk]
                            
                            try:
                                client.upload_df_to_bq(
                                    df=df_ods_conf_pdk,
                                    dataset=f"{NOM_LIB_ODS_C}.{conf_level}",
                                    table_name=NOM_TAB_ODS,
                                    write_disposition='WRITE_TRUNCATE'
                                )
                                
                                # Count columns in created table
                                try:
                                    query_count_cols_pdk = f"""
                                        SELECT COUNT(*) as NB_COL
                                        FROM `{project_id}.{NOM_LIB_ODS_C}.{conf_level}.INFORMATION_SCHEMA.COLUMNS`
                                        WHERE table_name = '{NOM_TAB_ODS}'
                                    """
                                    df_col_count_pdk = client.query(query_count_cols_pdk).to_dataframe()
                                    NB_COL_TABLI = int(df_col_count_pdk.iloc[0]['NB_COL'])
                                    
                                    # Delete table if it only contains base columns
                                    if NB_COL_TABLI == SIZE_MIN_BASE:
                                        logger.info(f"Deleting PDK confidential table {NOM_LIB_ODS_C}.{conf_level}.{NOM_TAB_ODS} - contains no confidential data")
                                        try:
                                            client.delete_table(f"{NOM_LIB_ODS_C}.{conf_level}", NOM_TAB_ODS, project_id)
                                        except Exception as e:
                                            logger.warning(f"Error deleting empty PDK confidential table: {str(e)}")
                                except Exception as e:
                                    logger.warning(f"Error counting columns in PDK confidential table level {conf_level}: {str(e)}")
                                    
                            except Exception as e:
                                logger.warning(f"Error creating PDK confidential table level {conf_level}: {str(e)}")
                
                # Count new keys
                NB_NEW_CLE_PDK = len(df_ods_pdk)
                
                # Log successful load
                today_dttm = dt.datetime.now()
                try:
                    util_dec_aco_alim_suivi_trt(
                        ID_TRT=MV_ID_TRT,
                        DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                        NOM_TRT="CHARGEMENT_ODS",
                        DAT_REEL_TRT=today_dttm,
                        MESG=f"La table {NOM_TAB_TAMPON} est chargé dans ODS : {NB_NEW_CLE_PDK} NOUVELLES CLES.",
                        CD_RETOUR=0
                    )
                except Exception as e:
                    logger.error(f"Error logging PDK table load: {str(e)}")
                    raise
                
                # Update tracking table
                try:
                    query_update_tracking_pdk = f"""
                        UPDATE `{project_id}.{lib_dec_aco_suivi}.ACO_SUIVI_CHG_TAMPON_ODS`
                        SET FLG_TRAITE_ODS = 'O',
                            CD_STATUT_ODS = 'C',
                            DAT_REEL_TRT_ODS = TIMESTAMP('{today_dttm}')
                        WHERE ID_TRT = {MV_ID_TRT}
                            AND DAT_BATCH_TRT = {MV_DAT_BATCH_TRT}
                            AND ID_FIC = '{ID_FIC}'
                    """
                    client.query(query_update_tracking_pdk)
                except Exception as e:
                    logger.error(f"Error updating tracking table for PDK: {str(e)}")
                    raise
                
                # Get MV_EXTENSION from params
                MV_EXTENSION = params.get("MV_EXTENSION", "")
                
                # Move file to archives directory
                try:
                    source_path = f"{REF_FIC_ENT}/ok/{NOM_FIC}"
                    dest_path = f"{REF_FIC_ENT}/archives/{NOM_FIC}.{MV_EXTENSION}"
                    
                    # Use GCS operations if paths are GCS URIs
                    if source_path.startswith("gs://"):
                        # Extract bucket and paths
                        source_parts = source_path.replace("gs://", "").split("/", 1)
                        dest_parts = dest_path.replace("gs://", "").split("/", 1)
                        
                        source_bucket = source_parts[0]
                        source_blob = source_parts[1]
                        dest_bucket = dest_parts[0]
                        dest_blob = dest_parts[1]
                        
                        # Move file in GCS
                        from google.cloud import storage
                        storage_client_gcs = storage.Client(project=project_id)
                        source_bucket_obj = storage_client_gcs.bucket(source_bucket)
                        source_blob_obj = source_bucket_obj.blob(source_blob)
                        dest_bucket_obj = storage_client_gcs.bucket(dest_bucket)
                        
                        source_bucket_obj.copy_blob(source_blob_obj, dest_bucket_obj, dest_blob)
                        source_blob_obj.delete()
                        
                        move_success = True
                    else:
                        # Use local file operations
                        import shutil
                        shutil.move(source_path, dest_path)
                        move_success = True
                except Exception as e:
                    logger.warning(f"Error moving PDK file {NOM_FIC}: {str(e)}")
                    move_success = False
                
                if move_success:
                    today_dttm = dt.datetime.now()
                    try:
                        util_dec_aco_alim_suivi_trt(
                            ID_TRT=MV_ID_TRT,
                            DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                            NOM_TRT="CHARGEMENT_ODS",
                            DAT_REEL_TRT=today_dttm,
                            MESG=f"Le fichier {NOM_FIC} a été déplacé dans le répertoire {REF_FIC_ENT}/archives.",
                            CD_RETOUR=0
                        )
                    except Exception as e:
                        logger.error(f"Error logging PDK file move: {str(e)}")
                    
                    # Zip the file
                    try:
                        archive_file = f"{NOM_FIC}.{MV_EXTENSION}"
                        archive_path = f"{REF_FIC_ENT}/archives/{archive_file}"
                        zip_path = f"{REF_FIC_ENT}/archives/Archives_{MV_EXTENSION}.zip"
                        
                        if archive_path.startswith("gs://"):
                            # For GCS, download, zip locally, upload
                            import tempfile
                            import zipfile
                            
                            with tempfile.TemporaryDirectory() as tmpdir:
                                # Download file
                                archive_parts = archive_path.replace("gs://", "").split("/", 1)
                                archive_bucket = archive_parts[0]
                                archive_blob = archive_parts[1]
                                
                                local_file = f"{tmpdir}/{archive_file}"
                                storage_client_gcs = storage.Client(project=project_id)
                                bucket_obj = storage_client_gcs.bucket(archive_bucket)
                                blob_obj = bucket_obj.blob(archive_blob)
                                blob_obj.download_to_filename(local_file)
                                
                                # Create or update zip
                                local_zip = f"{tmpdir}/Archives_{MV_EXTENSION}.zip"
                                
                                # Download existing zip if it exists
                                zip_parts = zip_path.replace("gs://", "").split("/", 1)
                                zip_blob = zip_parts[1]
                                zip_blob_obj = bucket_obj.blob(zip_blob)
                                
                                if zip_blob_obj.exists():
                                    zip_blob_obj.download_to_filename(local_zip)
                                    mode = 'a'
                                else:
                                    mode = 'w'
                                
                                # Add file to zip
                                with zipfile.ZipFile(local_zip, mode) as zf:
                                    zf.write(local_file, archive_file)
                                
                                # Upload zip back
                                zip_blob_obj.upload_from_filename(local_zip)
                                
                                zip_success = True
                        else:
                            # Local file system
                            import os
                            import zipfile
                            
                            mode = 'a' if os.path.exists(zip_path) else 'w'
                            with zipfile.ZipFile(zip_path, mode) as zf:
                                zf.write(archive_path, archive_file)
                            
                            zip_success = True
                    except Exception as e:
                        logger.warning(f"Error zipping PDK file {NOM_FIC}: {str(e)}")
                        zip_success = False
                    
                    if zip_success:
                        today_dttm = dt.datetime.now()
                        try:
                            util_dec_aco_alim_suivi_trt(
                                ID_TRT=MV_ID_TRT,
                                DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                                NOM_TRT="CHARGEMENT_ODS",
                                DAT_REEL_TRT=today_dttm,
                                MESG=f"Le fichier {NOM_FIC} a été ajouté à l'archive Archives_{MV_EXTENSION}.zip.",
                                CD_RETOUR=0
                            )
                        except Exception as e:
                            logger.error(f"Error logging PDK zip success: {str(e)}")
                        
                        # Remove the unzipped file
                        try:
                            if archive_path.startswith("gs://"):
                                archive_parts = archive_path.replace("gs://", "").split("/", 1)
                                archive_bucket = archive_parts[0]
                                archive_blob = archive_parts[1]
                                
                                storage_client_gcs = storage.Client(project=project_id)
                                bucket_obj = storage_client_gcs.bucket(archive_bucket)
                                blob_obj = bucket_obj.blob(archive_blob)
                                blob_obj.delete()
                            else:
                                import os
                                os.remove(archive_path)
                        except Exception as e:
                            logger.warning(f"Error removing archived PDK file: {str(e)}")
                    else:
                        today_dttm = dt.datetime.now()
                        try:
                            util_dec_aco_alim_suivi_trt(
                                ID_TRT=MV_ID_TRT,
                                DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                                NOM_TRT="CHARGEMENT_ODS",
                                DAT_REEL_TRT=today_dttm,
                                MESG=f"Impossible d'ajouter le fichier {NOM_FIC} à l'archive Archives_{MV_EXTENSION}.zip.",
                                CD_RETOUR=99
                            )
                        except Exception as e:
                            logger.error(f"Error logging PDK zip failure: {str(e)}")
                else:
                    today_dttm = dt.datetime.now()
                    try:
                        util_dec_aco_alim_suivi_trt(
                            ID_TRT=MV_ID_TRT,
                            DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                            NOM_TRT="CHARGEMENT_ODS",
                            DAT_REEL_TRT=today_dttm,
                            MESG=f"Le fichier {NOM_FIC} n'a pas pu être déplacé dans le répertoire {REF_FIC_ENT}/archives.",
                            CD_RETOUR=0
                        )
                    except Exception as e:
                        logger.error(f"Error logging PDK move failure: {str(e)}")
            
            # ELSE case: ODS table already exists - perform historization for PDK_SIEG
            else:
                logger.info(f"PDK_SIEG ODS table {NOM_LIB_ODS}.{NOM_TAB_ODS} already exists, performing historization")
                
                # Determine DAT_DEB_VAL based on system
                if ID_SYS_GEST == "GFP":
                    DAT_DEB_VAL = params.get(f"MV_DAT_ARRETE_{ID_SYS_GEST}")
                else:
                    DAT_DEB_VAL = MV_DAT_BATCH_TRT
                
                today_dttm = dt.datetime.now()
                
                # Sort existing ODS table
                try:
                    query_sort_ods_pdk = f"""
                        CREATE OR REPLACE TABLE `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}_SORTED` AS
                        SELECT *
                        FROM `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}`
                        ORDER BY ID_LIGNE, DT_ARRETE, DAT_DEB_VAL, DAT_FIN_VAL, NUM_SEQ, COD_ETA
                    """
                    client.query(query_sort_ods_pdk)
                except Exception as e:
                    logger.error(f"Error sorting PDK ODS table: {str(e)}")
                    raise
                
                # Load existing active records
                try:
                    query_active_pdk = f"""
                        CREATE OR REPLACE TABLE `{project_id}.WORK._ALL_DONNEES_{i}` AS
                        SELECT *
                        FROM `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}_SORTED`
                        WHERE COD_ETA = 1
                        ORDER BY ID_LIGNE, DT_ARRETE, DAT_DEB_VAL, DAT_FIN_VAL, NUM_SEQ, COD_ETA
                    """
                    client.query(query_active_pdk)
                except Exception as e:
                    logger.error(f"Error loading active PDK records: {str(e)}")
                    raise
                
                # Upload new data to work table
                df_ods_pdk['DAT_TRT'] = today_dttm
                df_ods_pdk['DAT_DEB_VAL'] = DAT_DEB_VAL
                df_ods_pdk['DAT_FIN_VAL'] = pd.to_datetime('9999-12-31')
                df_ods_pdk['DAT_DERN_RECPT'] = DAT_DEB_VAL
                
                try:
                    client.upload_df_to_bq(
                        df=df_ods_pdk,
                        dataset="WORK",
                        table_name=f"_{NOM_TAB_ODS}_{i}_NEW",
                        write_disposition='WRITE_TRUNCATE'
                    )
                except Exception as e:
                    logger.error(f"Error uploading new PDK data: {str(e)}")
                    raise
                
                # Sort new data
                try:
                    query_sort_new_pdk = f"""
                        CREATE OR REPLACE TABLE `{project_id}.WORK._{NOM_TAB_ODS}_{i}_SORTED` AS
                        SELECT *
                        FROM `{project_id}.WORK._{NOM_TAB_ODS}_{i}_NEW`
                        ORDER BY ID_LIGNE, DT_ARRETE
                    """
                    client.query(query_sort_new_pdk)
                except Exception as e:
                    logger.error(f"Error sorting new PDK data: {str(e)}")
                    raise
                
                # Identify new vs existing keys using merge
                try:
                    # Calculate last day of previous month for comparison
                    batch_date = dt.datetime.strptime(str(MV_DAT_BATCH_TRT), '%Y%m%d')
                    first_day_current_month = batch_date.replace(day=1)
                    last_day_prev_month = first_day_current_month - relativedelta(days=1)
                    last_day_prev_month_str = last_day_prev_month.strftime('%Y-%m-%d')
                    
                    query_merge_pdk = f"""
                        CREATE OR REPLACE TABLE `{project_id}.WORK._NOUVELLES_CLES{i}` AS
                        SELECT 
                            n.*,
                            CASE 
                                WHEN n.DT_ARRETE = DATE('{last_day_prev_month_str}') THEN 1
                                ELSE 0
                            END AS COD_ETA,
                            1 AS NUM_SEQ
                        FROM `{project_id}.WORK._{NOM_TAB_ODS}_{i}_SORTED` n
                        LEFT JOIN `{project_id}.WORK._ALL_DONNEES_{i}` e
                            ON n.ID_LIGNE = e.ID_LIGNE
                            AND n.DT_ARRETE = e.DT_ARRETE
                        WHERE e.ID_LIGNE IS NULL
                    """
                    client.query(query_merge_pdk)
                    
                    query_anciennes_pdk = f"""
                        CREATE OR REPLACE TABLE `{project_id}.WORK._ANCIENNES_CLES{i}` AS
                        SELECT 
                            n.*,
                            CASE 
                                WHEN n.DT_ARRETE = DATE('{last_day_prev_month_str}') THEN 1
                                ELSE 0
                            END AS COD_ETA
                        FROM `{project_id}.WORK._{NOM_TAB_ODS}_{i}_SORTED` n
                        INNER JOIN `{project_id}.WORK._ALL_DONNEES_{i}` e
                            ON n.ID_LIGNE = e.ID_LIGNE
                            AND n.DT_ARRETE = e.DT_ARRETE
                    """
                    client.query(query_anciennes_pdk)
                except Exception as e:
                    logger.error(f"Error identifying new/existing PDK keys: {str(e)}")
                    raise
                
                # Get NOM_COURT
                try:
                    query_nom_court_pdk = f"""
                        SELECT TYP_STRUCTURE_FIC
                        FROM `{project_id}.S_DACPRM.ACO_TAB_ODS`
                        WHERE NOM_TAB_TAMPON = '{NOM_TAB_TAMPON}'
                        LIMIT 1
                    """
                    df_nom_court_pdk = client.query(query_nom_court_pdk).to_dataframe()
                    NOM_COURT = df_nom_court_pdk.iloc[0]['TYP_STRUCTURE_FIC'] if not df_nom_court_pdk.empty else TYP_TAB_ODS
                except Exception as e:
                    logger.warning(f"Error getting NOM_COURT for PDK: {str(e)}")
                    NOM_COURT = TYP_TAB_ODS
                
                # Get comparable columns (excluding those marked for exclusion)
                try:
                    query_all_cle_comp_pdk = f"""
                        SELECT NOM_CHAMP
                        FROM `{project_id}.WORK.STRUCTURE_TABLE`
                        WHERE TYP_STRUCTURE_FIC = '{NOM_COURT}'
                            AND FLG_EXCLUSION_DIFF = 'N'
                        ORDER BY POSITION_CHAMP
                    """
                    df_all_cle_comp_pdk = client.query(query_all_cle_comp_pdk).to_dataframe()
                    ALL_CLE_COMP_t = ', '.join(df_all_cle_comp_pdk['NOM_CHAMP'].tolist())
                except Exception as e:
                    logger.error(f"Error getting comparable columns for PDK: {str(e)}")
                    raise
                
                logger.info(f"PDK comparable columns: {ALL_CLE_COMP_t}")
                
                # Find differences in old keys (records that changed)
                try:
                    query_diff_pdk = f"""
                        CREATE OR REPLACE TABLE `{project_id}.WORK.DIFF_OLD_CLE_TO_INSERT{i}` AS
                        SELECT {ALL_CLE_COMP_t}, ID_LIGNE, DT_ARRETE
                        FROM `{project_id}.WORK._ANCIENNES_CLES{i}`
                        EXCEPT DISTINCT
                        SELECT {ALL_CLE_COMP_t}, ID_LIGNE, DT_ARRETE
                        FROM `{project_id}.WORK._ALL_DONNEES_{i}`
                    """
                    client.query(query_diff_pdk)
                except Exception as e:
                    logger.error(f"Error finding PDK differences: {str(e)}")
                    raise
                
                # Sort for merge
                try:
                    query_sort_anciennes_pdk = f"""
                        CREATE OR REPLACE TABLE `{project_id}.WORK._ANCIENNES_CLES{i}_SORTED` AS
                        SELECT *
                        FROM `{project_id}.WORK._ANCIENNES_CLES{i}`
                        ORDER BY ID_LIGNE, DT_ARRETE
                    """
                    client.query(query_sort_anciennes_pdk)
                    
                    query_sort_diff_pdk = f"""
                        CREATE OR REPLACE TABLE `{project_id}.WORK.DIFF_OLD_CLE_TO_INSERT{i}_SORTED` AS
                        SELECT *
                        FROM `{project_id}.WORK.DIFF_OLD_CLE_TO_INSERT{i}`
                        ORDER BY ID_LIGNE, DT_ARRETE
                    """
                    client.query(query_sort_diff_pdk)
                except Exception as e:
                    logger.error(f"Error sorting PDK tables for merge: {str(e)}")
                    raise
                
                # Identify records to insert (changed) and records to update DAT_DERN_RECPT (unchanged)
                try:
                    query_old_insert_pdk = f"""
                        CREATE OR REPLACE TABLE `{project_id}.WORK._OLD_CLE_TO_INSERT{i}` AS
                        SELECT a.*
                        FROM `{project_id}.WORK._ANCIENNES_CLES{i}_SORTED` a
                        INNER JOIN `{project_id}.WORK.DIFF_OLD_CLE_TO_INSERT{i}_SORTED` b
                            ON a.ID_LIGNE = b.ID_LIGNE
                            AND a.DT_ARRETE = b.DT_ARRETE
                    """
                    client.query(query_old_insert_pdk)
                    
                    query_maj_date_pdk = f"""
                        CREATE OR REPLACE TABLE `{project_id}.WORK._MAJ_DATE_RECP{i}` AS
                        SELECT a.ID_LIGNE, a.DT_ARRETE
                        FROM `{project_id}.WORK._ANCIENNES_CLES{i}_SORTED` a
                        LEFT JOIN `{project_id}.WORK.DIFF_OLD_CLE_TO_INSERT{i}_SORTED` b
                            ON a.ID_LIGNE = b.ID_LIGNE
                            AND a.DT_ARRETE = b.DT_ARRETE
                        WHERE b.ID_LIGNE IS NULL
                    """
                    client.query(query_maj_date_pdk)
                except Exception as e:
                    logger.error(f"Error identifying PDK changed/unchanged records: {str(e)}")
                    raise
                
                # Prepare old keys for insertion into main table
                try:
                    # Select columns for main ODS table (non-confidential or key columns)
                    cols_main_old_pdk = ['ID_LIGNE', 'DT_ARRETE', 'DAT_TRT', 'DAT_DEB_VAL', 'DAT_FIN_VAL', 'COD_ETA', 'NUM_SEQ', 'DAT_DERN_RECPT']
                    for field_name, info in field_info.items():
                        if info['flg_cle'] == 'O' or info['flg_conf'] == 'N' or info['flg_all_lib'] == 'O':
                            cols_main_old_pdk.append(field_name)
                    
                    cols_main_old_str_pdk = ', '.join(cols_main_old_pdk)
                    
                    query_prep_old_main_pdk = f"""
                        CREATE OR REPLACE TABLE `{project_id}.WORK._OLD_CLE_TO_INSERT{i}_MAIN` AS
                        SELECT {cols_main_old_str_pdk}
                        FROM `{project_id}.WORK._OLD_CLE_TO_INSERT{i}`
                    """
                    client.query(query_prep_old_main_pdk)
                except Exception as e:
                    logger.error(f"Error preparing old PDK keys for main table: {str(e)}")
                    raise
                
                # Prepare old keys for confidential tables
                if NOM_LIB_ODS_C and NOM_LIB_ODS_C.strip():
                    for conf_level in range(1, 5):
                        try:
                            # Select columns for this confidentiality level
                            cols_conf_old_pdk = ['ID_LIGNE', 'DT_ARRETE', 'DAT_TRT', 'DAT_DEB_VAL', 'DAT_FIN_VAL', 'COD_ETA', 'NUM_SEQ', 'DAT_DERN_RECPT']
                            for field_name, info in field_info.items():
                                if (info['flg_cle'] == 'O' or 
                                    info['flg_all_lib'] == 'O' or 
                                    (info['flg_conf'] == 'O' and info['niv_conf'] == conf_level)):
                                    cols_conf_old_pdk.append(field_name)
                            
                            cols_conf_old_str_pdk = ', '.join(cols_conf_old_pdk)
                            
                            query_prep_old_conf_pdk = f"""
                                CREATE OR REPLACE TABLE `{project_id}.WORK._OLD_CLE_TO_INSERT{i}_CONF{conf_level}` AS
                                SELECT {cols_conf_old_str_pdk}
                                FROM `{project_id}.WORK._OLD_CLE_TO_INSERT{i}`
                            """
                            client.query(query_prep_old_conf_pdk)
                        except Exception as e:
                            logger.warning(f"Error preparing old PDK keys for confidential table level {conf_level}: {str(e)}")
                
                # Update DAT_DERN_RECPT for unchanged records in main ODS table
                try:
                    query_update_dat_dern_pdk = f"""
                        UPDATE `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}` o
                        SET DAT_DERN_RECPT = DATE('{DAT_DEB_VAL}')
                        WHERE EXISTS (
                            SELECT 1
                            FROM `{project_id}.WORK._MAJ_DATE_RECP{i}` m
                            WHERE o.ID_LIGNE = m.ID_LIGNE
                                AND o.DT_ARRETE = m.DT_ARRETE
                        )
                    """
                    client.query(query_update_dat_dern_pdk)
                except Exception as e:
                    logger.error(f"Error updating DAT_DERN_RECPT in PDK main ODS table: {str(e)}")
                    raise
                
                # Update DAT_DERN_RECPT for unchanged records in confidential tables
                if NOM_LIB_ODS_C and NOM_LIB_ODS_C.strip():
                    for conf_level in range(1, 5):
                        try:
                            # Check if confidential table exists
                            conf_table_exists = client.table_exists(f"{NOM_LIB_ODS_C}.{conf_level}", NOM_TAB_ODS)
                            
                            if conf_table_exists:
                                query_update_dat_dern_conf_pdk = f"""
                                    UPDATE `{project_id}.{NOM_LIB_ODS_C}.{conf_level}.{NOM_TAB_ODS}` o
                                    SET DAT_DERN_RECPT = DATE('{DAT_DEB_VAL}')
                                    WHERE EXISTS (
                                        SELECT 1
                                        FROM `{project_id}.WORK._MAJ_DATE_RECP{i}` m
                                        WHERE o.ID_LIGNE = m.ID_LIGNE
                                            AND o.DT_ARRETE = m.DT_ARRETE
                                    )
                                """
                                client.query(query_update_dat_dern_conf_pdk)
                        except Exception as e:
                            logger.warning(f"Error updating DAT_DERN_RECPT in PDK confidential table level {conf_level}: {str(e)}")
                
                # Append new keys to main ODS table
                try:
                    # Select columns for main ODS table
                    cols_main_new_pdk = ['ID_LIGNE', 'DT_ARRETE', 'DAT_TRT', 'DAT_DEB_VAL', 'DAT_FIN_VAL', 'COD_ETA', 'NUM_SEQ', 'DAT_DERN_RECPT']
                    for field_name, info in field_info.items():
                        if info['flg_cle'] == 'O' or info['flg_conf'] == 'N' or info['flg_all_lib'] == 'O':
                            cols_main_new_pdk.append(field_name)
                    
                    cols_main_new_str_pdk = ', '.join(cols_main_new_pdk)
                    
                    query_append_new_main_pdk = f"""
                        INSERT INTO `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}`
                        SELECT {cols_main_new_str_pdk}
                        FROM `{project_id}.WORK._NOUVELLES_CLES{i}`
                    """
                    client.query(query_append_new_main_pdk)
                except Exception as e:
                    logger.error(f"Error appending new PDK keys to main ODS table: {str(e)}")
                    raise
                
                # Append changed keys (old keys to insert) to main ODS table
                try:
                    query_append_old_main_pdk = f"""
                        INSERT INTO `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}`
                        SELECT *
                        FROM `{project_id}.WORK._OLD_CLE_TO_INSERT{i}_MAIN`
                    """
                    client.query(query_append_old_main_pdk)
                except Exception as e:
                    logger.error(f"Error appending changed PDK keys to main ODS table: {str(e)}")
                    raise
                
                # Append new and changed keys to confidential tables
                if NOM_LIB_ODS_C and NOM_LIB_ODS_C.strip():
                    for conf_level in range(1, 5):
                        try:
                            # Check if confidential table exists
                            conf_table_exists = client.table_exists(f"{NOM_LIB_ODS_C}.{conf_level}", NOM_TAB_ODS)
                            
                            if conf_table_exists:
                                # Select columns for this confidentiality level
                                cols_conf_new_pdk = ['ID_LIGNE', 'DT_ARRETE', 'DAT_TRT', 'DAT_DEB_VAL', 'DAT_FIN_VAL', 'COD_ETA', 'NUM_SEQ', 'DAT_DERN_RECPT']
                                for field_name, info in field_info.items():
                                    if (info['flg_cle'] == 'O' or 
                                        info['flg_all_lib'] == 'O' or 
                                        (info['flg_conf'] == 'O' and info['niv_conf'] == conf_level)):
                                        cols_conf_new_pdk.append(field_name)
                                
                                cols_conf_new_str_pdk = ', '.join(cols_conf_new_pdk)
                                
                                # Append new keys
                                query_append_new_conf_pdk = f"""
                                    INSERT INTO `{project_id}.{NOM_LIB_ODS_C}.{conf_level}.{NOM_TAB_ODS}`
                                    SELECT {cols_conf_new_str_pdk}
                                    FROM `{project_id}.WORK._NOUVELLES_CLES{i}`
                                """
                                client.query(query_append_new_conf_pdk)
                                
                                # Append changed keys
                                query_append_old_conf_pdk = f"""
                                    INSERT INTO `{project_id}.{NOM_LIB_ODS_C}.{conf_level}.{NOM_TAB_ODS}`
                                    SELECT *
                                    FROM `{project_id}.WORK._OLD_CLE_TO_INSERT{i}_CONF{conf_level}`
                                """
                                client.query(query_append_old_conf_pdk)
                        except Exception as e:
                            logger.warning(f"Error appending PDK keys to confidential table level {conf_level}: {str(e)}")
                
                # Update DAT_FIN_VAL using window functions to close out previous versions in main table
                try:
                    # Get key columns for partitioning
                    key_cols_pdk = []
                    for field_name, info in field_info.items():
                        if info['flg_cle'] == 'O':
                            key_cols_pdk.append(field_name)
                    
                    key_cols_str_pdk = ', '.join(key_cols_pdk)
                    
                    query_update_dat_fin_pdk = f"""
                        CREATE OR REPLACE TABLE `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}` AS
                        SELECT
                            * EXCEPT(DAT_FIN_VAL),
                            COALESCE(
                                DATE_SUB(
                                    LEAD(DAT_DEB_VAL) OVER (
                                        PARTITION BY {key_cols_str_pdk}, DT_ARRETE
                                        ORDER BY DAT_DEB_VAL, NUM_SEQ
                                    ),
                                    INTERVAL 1 DAY
                                ),
                                DATE('9999-12-31')
                            ) AS DAT_FIN_VAL
                        FROM `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}`
                    """
                    client.query(query_update_dat_fin_pdk)
                except Exception as e:
                    logger.error(f"Error updating DAT_FIN_VAL in PDK main ODS table: {str(e)}")
                    raise
                
                # Update DAT_FIN_VAL in confidential tables
                if NOM_LIB_ODS_C and NOM_LIB_ODS_C.strip():
                    for conf_level in range(1, 5):
                        try:
                            # Check if confidential table exists
                            conf_table_exists = client.table_exists(f"{NOM_LIB_ODS_C}.{conf_level}", NOM_TAB_ODS)
                            
                            if conf_table_exists:
                                query_update_dat_fin_conf_pdk = f"""
                                    CREATE OR REPLACE TABLE `{project_id}.{NOM_LIB_ODS_C}.{conf_level}.{NOM_TAB_ODS}` AS
                                    SELECT
                                        * EXCEPT(DAT_FIN_VAL),
                                        COALESCE(
                                            DATE_SUB(
                                                LEAD(DAT_DEB_VAL) OVER (
                                                    PARTITION BY {key_cols_str_pdk}, DT_ARRETE
                                                    ORDER BY DAT_DEB_VAL, NUM_SEQ
                                                ),
                                                INTERVAL 1 DAY
                                            ),
                                            DATE('9999-12-31')
                                        ) AS DAT_FIN_VAL
                                    FROM `{project_id}.{NOM_LIB_ODS_C}.{conf_level}.{NOM_TAB_ODS}`
                                """
                                client.query(query_update_dat_fin_conf_pdk)
                        except Exception as e:
                            logger.warning(f"Error updating DAT_FIN_VAL in PDK confidential table level {conf_level}: {str(e)}")
                
                # Recalculate NUM_SEQ and COD_ETA in main ODS table
                try:
                    query_recalc_seq_eta_pdk = f"""
                        CREATE OR REPLACE TABLE `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}` AS
                        SELECT
                            * EXCEPT(NUM_SEQ, COD_ETA),
                            ROW_NUMBER() OVER (
                                PARTITION BY {key_cols_str_pdk}, DT_ARRETE
                                ORDER BY DAT_DEB_VAL
                            ) AS NUM_SEQ,
                            CASE
                                WHEN DAT_FIN_VAL = DATE('9999-12-31') THEN 1
                                ELSE 0
                            END AS COD_ETA
                        FROM `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}`
                        ORDER BY ID_LIGNE, DT_ARRETE, DAT_DEB_VAL, DAT_FIN_VAL, NUM_SEQ, COD_ETA
                    """
                    client.query(query_recalc_seq_eta_pdk)
                except Exception as e:
                    logger.error(f"Error recalculating NUM_SEQ and COD_ETA in PDK main ODS table: {str(e)}")
                    raise
                
                # Recalculate NUM_SEQ and COD_ETA in confidential tables
                if NOM_LIB_ODS_C and NOM_LIB_ODS_C.strip():
                    for conf_level in range(1, 5):
                        try:
                            # Check if confidential table exists
                            conf_table_exists = client.table_exists(f"{NOM_LIB_ODS_C}.{conf_level}", NOM_TAB_ODS)
                            
                            if conf_table_exists:
                                query_recalc_seq_eta_conf_pdk = f"""
                                    CREATE OR REPLACE TABLE `{project_id}.{NOM_LIB_ODS_C}.{conf_level}.{NOM_TAB_ODS}` AS
                                    SELECT
                                        * EXCEPT(NUM_SEQ, COD_ETA),
                                        ROW_NUMBER() OVER (
                                            PARTITION BY {key_cols_str_pdk}, DT_ARRETE
                                            ORDER BY DAT_DEB_VAL
                                        ) AS NUM_SEQ,
                                        CASE
                                            WHEN DAT_FIN_VAL = DATE('9999-12-31') THEN 1
                                            ELSE 0
                                        END AS COD_ETA
                                    FROM `{project_id}.{NOM_LIB_ODS_C}.{conf_level}.{NOM_TAB_ODS}`
                                    ORDER BY ID_LIGNE, DT_ARRETE, DAT_DEB_VAL, DAT_FIN_VAL, NUM_SEQ, COD_ETA
                                """
                                client.query(query_recalc_seq_eta_conf_pdk)
                        except Exception as e:
                            logger.warning(f"Error recalculating NUM_SEQ and COD_ETA in PDK confidential table level {conf_level}: {str(e)}")
                
                # Count new and old keys
                try:
                    query_count_new_pdk = f"""
                        SELECT COUNT(*) AS NB_NEW_CLE
                        FROM `{project_id}.WORK._NOUVELLES_CLES{i}`
                    """
                    df_count_new_pdk = client.query(query_count_new_pdk).to_dataframe()
                    NB_NEW_CLE_PDK = int(df_count_new_pdk.iloc[0]['NB_NEW_CLE'])
                    
                    query_count_old_pdk = f"""
                        SELECT COUNT(*) AS NB_OLD_CLE
                        FROM `{project_id}.WORK._OLD_CLE_TO_INSERT{i}`
                    """
                    df_count_old_pdk = client.query(query_count_old_pdk).to_dataframe()
                    NB_OLD_CLE_PDK = int(df_count_old_pdk.iloc[0]['NB_OLD_CLE'])
                except Exception as e:
                    logger.error(f"Error counting PDK new/old keys: {str(e)}")
                    raise
                
                # Log successful historization
                today_dttm = dt.datetime.now()
                try:
                    util_dec_aco_alim_suivi_trt(
                        ID_TRT=MV_ID_TRT,
                        DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                        NOM_TRT="CHARGEMENT_ODS",
                        DAT_REEL_TRT=today_dttm,
                        MESG=f"La table {NOM_TAB_TAMPON} est chargé dans ODS : {NB_NEW_CLE_PDK} NOUVELLES CLES et {NB_OLD_CLE_PDK} ANCIENNES CLES MODIFIEES.",
                        CD_RETOUR=0
                    )
                except Exception as e:
                    logger.error(f"Error logging PDK historization: {str(e)}")
                    raise
                
                # Update tracking table
                try:
                    query_update_tracking_hist_pdk = f"""
                        UPDATE `{project_id}.{lib_dec_aco_suivi}.ACO_SUIVI_CHG_TAMPON_ODS`
                        SET FLG_TRAITE_ODS = 'O',
                            CD_STATUT_ODS = 'C',
                            DAT_REEL_TRT_ODS = TIMESTAMP('{today_dttm}')
                        WHERE ID_TRT = {MV_ID_TRT}
                            AND DAT_BATCH_TRT = {MV_DAT_BATCH_TRT}
                            AND ID_FIC = '{ID_FIC}'
                    """
                    client.query(query_update_tracking_hist_pdk)
                except Exception as e:
                    logger.error(f"Error updating tracking table for PDK historization: {str(e)}")
                    raise
                
                # Get MV_EXTENSION from params
                MV_EXTENSION = params.get("MV_EXTENSION", "")
                
                # Move file to archives directory
                try:
                    source_path = f"{REF_FIC_ENT}/ok/{NOM_FIC}"
                    dest_path = f"{REF_FIC_ENT}/archives/{NOM_FIC}.{MV_EXTENSION}"
                    
                    # Use GCS operations if paths are GCS URIs
                    if source_path.startswith("gs://"):
                        # Extract bucket and paths
                        source_parts = source_path.replace("gs://", "").split("/", 1)
                        dest_parts = dest_path.replace("gs://", "").split("/", 1)
                        
                        source_bucket = source_parts[0]
                        source_blob = source_parts[1]
                        dest_bucket = dest_parts[0]
                        dest_blob = dest_parts[1]
                        
                        # Move file in GCS
                        from google.cloud import storage
                        storage_client_gcs = storage.Client(project=project_id)
                        source_bucket_obj = storage_client_gcs.bucket(source_bucket)
                        source_blob_obj = source_bucket_obj.blob(source_blob)
                        dest_bucket_obj = storage_client_gcs.bucket(dest_bucket)
                        
                        source_bucket_obj.copy_blob(source_blob_obj, dest_bucket_obj, dest_blob)
                        source_blob_obj.delete()
                        
                        move_success = True
                    else:
                        # Use local file operations
                        import shutil
                        shutil.move(source_path, dest_path)
                        move_success = True
                except Exception as e:
                    logger.warning(f"Error moving PDK file {NOM_FIC} (historization): {str(e)}")
                    move_success = False
                
                if move_success:
                    today_dttm = dt.datetime.now()
                    try:
                        util_dec_aco_alim_suivi_trt(
                            ID_TRT=MV_ID_TRT,
                            DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                            NOM_TRT="CHARGEMENT_ODS",
                            DAT_REEL_TRT=today_dttm,
                            MESG=f"Le fichier {NOM_FIC} a été déplacé dans le répertoire {REF_FIC_ENT}/archives.",
                            CD_RETOUR=0
                        )
                    except Exception as e:
                        logger.error(f"Error logging PDK file move (historization): {str(e)}")
                    
                    # Zip the file
                    try:
                        archive_file = f"{NOM_FIC}.{MV_EXTENSION}"
                        archive_path = f"{REF_FIC_ENT}/archives/{archive_file}"
                        zip_path = f"{REF_FIC_ENT}/archives/Archives_{MV_EXTENSION}.zip"
                        
                        if archive_path.startswith("gs://"):
                            # For GCS, download, zip locally, upload
                            import tempfile
                            import zipfile
                            
                            with tempfile.TemporaryDirectory() as tmpdir:
                                # Download file
                                archive_parts = archive_path.replace("gs://", "").split("/", 1)
                                archive_bucket = archive_parts[0]
                                archive_blob = archive_parts[1]
                                
                                local_file = f"{tmpdir}/{archive_file}"
                                storage_client_gcs = storage.Client(project=project_id)
                                bucket_obj = storage_client_gcs.bucket(archive_bucket)
                                blob_obj = bucket_obj.blob(archive_blob)
                                blob_obj.download_to_filename(local_file)
                                
                                # Create or update zip
                                local_zip = f"{tmpdir}/Archives_{MV_EXTENSION}.zip"
                                
                                # Download existing zip if it exists
                                zip_parts = zip_path.replace("gs://", "").split("/", 1)
                                zip_blob = zip_parts[1]
                                zip_blob_obj = bucket_obj.blob(zip_blob)
                                
                                if zip_blob_obj.exists():
                                    zip_blob_obj.download_to_filename(local_zip)
                                    mode = 'a'
                                else:
                                    mode = 'w'
                                
                                # Add file to zip
                                with zipfile.ZipFile(local_zip, mode) as zf:
                                    zf.write(local_file, archive_file)
                                
                                # Upload zip back
                                zip_blob_obj.upload_from_filename(local_zip)
                                
                                zip_success = True
                        else:
                            # Local file system
                            import os
                            import zipfile
                            
                            mode = 'a' if os.path.exists(zip_path) else 'w'
                            with zipfile.ZipFile(zip_path, mode) as zf:
                                zf.write(archive_path, archive_file)
                            
                            zip_success = True
                    except Exception as e:
                        logger.warning(f"Error zipping PDK file {NOM_FIC} (historization): {str(e)}")
                        zip_success = False
                    
                    if zip_success:
                        today_dttm = dt.datetime.now()
                        try:
                            util_dec_aco_alim_suivi_trt(
                                ID_TRT=MV_ID_TRT,
                                DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                                NOM_TRT="CHARGEMENT_ODS",
                                DAT_REEL_TRT=today_dttm,
                                MESG=f"Le fichier {NOM_FIC} a été ajouté à l'archive Archives_{MV_EXTENSION}.zip.",
                                CD_RETOUR=0
                            )
                        except Exception as e:
                            logger.error(f"Error logging PDK zip success (historization): {str(e)}")
                        
                        # Remove the unzipped file
                        try:
                            if archive_path.startswith("gs://"):
                                archive_parts = archive_path.replace("gs://", "").split("/", 1)
                                archive_bucket = archive_parts[0]
                                archive_blob = archive_parts[1]
                                
                                storage_client_gcs = storage.Client(project=project_id)
                                bucket_obj = storage_client_gcs.bucket(archive_bucket)
                                blob_obj = bucket_obj.blob(archive_blob)
                                blob_obj.delete()
                            else:
                                import os
                                os.remove(archive_path)
                        except Exception as e:
                            logger.warning(f"Error removing archived PDK file (historization): {str(e)}")
                    else:
                        today_dttm = dt.datetime.now()
                        try:
                            util_dec_aco_alim_suivi_trt(
                                ID_TRT=MV_ID_TRT,
                                DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                                NOM_TRT="CHARGEMENT_ODS",
                                DAT_REEL_TRT=today_dttm,
                                MESG=f"Impossible d'ajouter le fichier {NOM_FIC} à l'archive Archives_{MV_EXTENSION}.zip.",
                                CD_RETOUR=99
                            )
                        except Exception as e:
                            logger.error(f"Error logging PDK zip failure (historization): {str(e)}")
                else:
                    today_dttm = dt.datetime.now()
                    try:
                        util_dec_aco_alim_suivi_trt(
                            ID_TRT=MV_ID_TRT,
                            DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                            NOM_TRT="CHARGEMENT_ODS",
                            DAT_REEL_TRT=today_dttm,
                            MESG=f"Le fichier {NOM_FIC} n'a pas pu être déplacé dans le répertoire {REF_FIC_ENT}/archives.",
                            CD_RETOUR=0
                        )
                    except Exception as e:
                        logger.error(f"Error logging PDK move failure (historization): {str(e)}")
        
        # ELSE case: Handle errors in PDK_SIEG validation
        else:
            logger.warning(f"PDK_SIEG table {NOM_TAB_TAMPON} has {NB_ERR_PDK} errors, moving to KO directory")
            
            # Count different error types
            try:
                NB_ERR_DATE_PDK = len(df_err_date_pdk)
                NB_ERR_CHAR_PDK = len(df_err_char_pdk)
                NB_ERR_NUM_PDK = len(df_err_num_pdk)
                NB_ERR_CLE_PDK = len(df_err_cle_pdk)
            except Exception as e:
                logger.error(f"Error counting PDK error types: {str(e)}")
                raise
            
            # Log error summary
            today_dttm = dt.datetime.now()
            try:
                util_dec_aco_alim_suivi_trt(
                    ID_TRT=MV_ID_TRT,
                    DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                    NOM_TRT="CHARGEMENT_ODS",
                    DAT_REEL_TRT=today_dttm,
                    MESG=f"Erreurs dans la table {NOM_TAB_TAMPON}: {NB_ERR_DATE_PDK} erreurs de date, {NB_ERR_CHAR_PDK} erreurs de caractère, {NB_ERR_NUM_PDK} erreurs numériques, {NB_ERR_CLE_PDK} erreurs de clé.",
                    CD_RETOUR=99
                )
            except Exception as e:
                logger.error(f"Error logging PDK error summary: {str(e)}")
                raise
            
            # Move file to KO directory
            try:
                source_path = f"{REF_FIC_ENT}/ok/{NOM_FIC}"
                dest_path = f"{REF_FIC_ENT}/ko/{NOM_FIC}"
                
                # Use GCS operations if paths are GCS URIs
                if source_path.startswith("gs://"):
                    # Extract bucket and paths
                    source_parts = source_path.replace("gs://", "").split("/", 1)
                    dest_parts = dest_path.replace("gs://", "").split("/", 1)
                    
                    source_bucket = source_parts[0]
                    source_blob = source_parts[1]
                    dest_bucket = dest_parts[0]
                    dest_blob = dest_parts[1]
                    
                    # Move file in GCS
                    from google.cloud import storage
                    storage_client_gcs = storage.Client(project=project_id)
                    source_bucket_obj = storage_client_gcs.bucket(source_bucket)
                    source_blob_obj = source_bucket_obj.blob(source_blob)
                    dest_bucket_obj = storage_client_gcs.bucket(dest_bucket)
                    
                    source_bucket_obj.copy_blob(source_blob_obj, dest_bucket_obj, dest_blob)
                    source_blob_obj.delete()
                    
                    move_ko_success = True
                else:
                    # Use local file operations
                    import shutil
                    shutil.move(source_path, dest_path)
                    move_ko_success = True
            except Exception as e:
                logger.warning(f"Error moving PDK file {NOM_FIC} to KO: {str(e)}")
                move_ko_success = False
            
            if move_ko_success:
                today_dttm = dt.datetime.now()
                try:
                    util_dec_aco_alim_suivi_trt(
                        ID_TRT=MV_ID_TRT,
                        DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                        NOM_TRT="CHARGEMENT_ODS",
                        DAT_REEL_TRT=today_dttm,
                        MESG=f"Le fichier {NOM_FIC} a été déplacé dans le répertoire {REF_FIC_ENT}/ko.",
                        CD_RETOUR=0
                    )
                except Exception as e:
                    logger.error(f"Error logging PDK file move to KO: {str(e)}")
            else:
                today_dttm = dt.datetime.now()
                try:
                    util_dec_aco_alim_suivi_trt(
                        ID_TRT=MV_ID_TRT,
                        DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                        NOM_TRT="CHARGEMENT_ODS",
                        DAT_REEL_TRT=today_dttm,
                        MESG=f"Le fichier {NOM_FIC} n'a pas pu être déplacé dans le répertoire {REF_FIC_ENT}/ko.",
                        CD_RETOUR=99
                    )
                except Exception as e:
                    logger.error(f"Error logging PDK move to KO failure: {str(e)}")
            
            # Update tracking table with error status
            try:
                query_update_tracking_err_pdk = f"""
                    UPDATE `{project_id}.{lib_dec_aco_suivi}.ACO_SUIVI_CHG_TAMPON_ODS`
                    SET FLG_TRAITE_ODS = 'O',
                        CD_STATUT_ODS = 'E',
                        DAT_REEL_TRT_ODS = TIMESTAMP('{today_dttm}')
                    WHERE ID_TRT = {MV_ID_TRT}
                        AND DAT_BATCH_TRT = {MV_DAT_BATCH_TRT}
                        AND ID_FIC = '{ID_FIC}'
                """
                client.query(query_update_tracking_err_pdk)
            except Exception as e:
                logger.error(f"Error updating tracking table for PDK errors: {str(e)}")
                raise

# Prepare _OLD_CLE_TO_INSERT with proper structure for PDK_SIEG
            try:
                # Calculate ID_LIGNE and DT_ARRETE
                # ID_LIGNE = _N_ (row number)
                # DT_ARRETE = last day of previous month
                
                # Build column list for main table
                cols_old_insert_pdk = []
                for field_name, info in field_info.items():
                    if info['flg_cle'] == 'O' or info['flg_conf'] == 'N' or info['flg_all_lib'] == 'O':
                        cols_old_insert_pdk.append(field_name)
                
                cols_old_insert_str = ', '.join(cols_old_insert_pdk)
                
                # Calculate last day of previous month
                query_prep_old_insert_pdk = f"""
                    CREATE OR REPLACE TABLE `{project_id}.WORK._OLD_CLE_TO_INSERT{i}` AS
                    SELECT
                        ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS ID_LIGNE,
                        LAST_DAY(DATE_SUB(DATE('{MV_DAT_BATCH_TRT}'), INTERVAL 1 MONTH)) AS DT_ARRETE,
                        TIMESTAMP('{today_dttm}') AS DAT_TRT,
                        DATE('{DAT_DEB_VAL}') AS DAT_DEB_VAL,
                        DATE('9999-12-31') AS DAT_FIN_VAL,
                        CASE 
                            WHEN LAST_DAY(DATE_SUB(DATE('{MV_DAT_BATCH_TRT}'), INTERVAL 1 MONTH)) = 
                                    LAST_DAY(DATE_SUB(DATE('{MV_DAT_BATCH_TRT}'), INTERVAL 1 MONTH))
                            THEN 1
                            ELSE 0
                        END AS COD_ETA,
                        1 AS NUM_SEQ,
                        DATE('{DAT_DEB_VAL}') AS DAT_DERN_RECPT,
                        {cols_old_insert_str}
                    FROM `{project_id}.WORK._OLD_CLE_TO_INSERT{i}`
                """
                client.query(query_prep_old_insert_pdk)
            except Exception as e:
                logger.error(f"Error preparing _OLD_CLE_TO_INSERT for PDK: {{str(e)}}")
                raise
            
            # Append new keys to main ODS table
            try:
                query_append_new_pdk = f"""
                    INSERT INTO `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}`
                    SELECT *
                    FROM `{project_id}.WORK._NOUVELLES_CLES{i}`
                """
                client.query(query_append_new_pdk)
            except Exception as e:
                logger.error(f"Error appending new PDK keys to main ODS table: {{str(e)}}")
                raise
                
            # Append old keys (changed records) to main ODS table
            try:
                query_append_old_pdk = f"""
                    INSERT INTO `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}`
                    SELECT *
                    FROM `{project_id}.WORK._OLD_CLE_TO_INSERT{i}`
                """
                client.query(query_append_old_pdk)
            except Exception as e:
                logger.error(f"Error appending changed PDK keys to main ODS table: {{str(e)}}")
                raise
            
            # Sort main ODS table by ID_LIGNE, DT_ARRETE descending DAT_DEB_VAL
            try:
                query_sort_desc_pdk = f"""
                    CREATE OR REPLACE TABLE `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}` AS
                    SELECT *
                    FROM `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}`
                    ORDER BY ID_LIGNE, DT_ARRETE, DAT_DEB_VAL DESC
                """
                client.query(query_sort_desc_pdk)
            except Exception as e:
                logger.error(f"Error sorting PDK ODS table descending: {{str(e)}}")
                raise
                
            # Update DAT_FIN_VAL using LAG function
            try:
                query_update_fin_val_pdk = f"""
                    CREATE OR REPLACE TABLE `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}` AS
                    WITH ranked_data AS (
                        SELECT
                            *,
                            LAG(DAT_DEB_VAL) OVER (
                                PARTITION BY ID_LIGNE, DT_ARRETE
                                ORDER BY DAT_DEB_VAL DESC
                            ) AS DAT_FIN_VAL2,
                            ROW_NUMBER() OVER (
                                PARTITION BY ID_LIGNE, DT_ARRETE
                                ORDER BY DAT_DEB_VAL DESC
                            ) AS rn
                        FROM `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}`
                    )
                    SELECT
                        * EXCEPT(DAT_FIN_VAL, DAT_FIN_VAL2, rn),
                        CASE
                            WHEN rn > 1 AND DAT_FIN_VAL = DATE('9999-12-31')
                            THEN DATE_SUB(DAT_FIN_VAL2, INTERVAL 1 DAY)
                            ELSE DAT_FIN_VAL
                        END AS DAT_FIN_VAL
                    FROM ranked_data
                """
                client.query(query_update_fin_val_pdk)
            except Exception as e:
                logger.error(f"Error updating DAT_FIN_VAL for PDK: {{str(e)}}")
                raise
                
            # Sort ascending by DAT_DEB_VAL
            try:
                query_sort_asc_pdk = f"""
                    CREATE OR REPLACE TABLE `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}` AS
                    SELECT *
                    FROM `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}`
                    ORDER BY ID_LIGNE, DT_ARRETE, DAT_DEB_VAL
                """
                client.query(query_sort_asc_pdk)
            except Exception as e:
                logger.error(f"Error sorting PDK ODS table ascending: {{str(e)}}")
                raise
                
            # Recalculate NUM_SEQ and set COD_ETA
            try:
                query_recalc_seq_pdk = f"""
                    CREATE OR REPLACE TABLE `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}` AS
                    SELECT
                        * EXCEPT(NUM_SEQ, COD_ETA),
                        ROW_NUMBER() OVER (
                            PARTITION BY ID_LIGNE, DT_ARRETE
                            ORDER BY DAT_DEB_VAL
                        ) AS NUM_SEQ,
                        1 AS COD_ETA
                    FROM `{project_id}.{NOM_LIB_ODS}.{NOM_TAB_ODS}`
                """
                client.query(query_recalc_seq_pdk)
            except Exception as e:
                logger.error(f"Error recalculating NUM_SEQ and COD_ETA for PDK: {{str(e)}}")
                raise
                
            # Get NOM_COURT for column list building
            try:
                query_nom_court_pdk = f"""
                    SELECT TYP_STRUCTURE_FIC
                    FROM `{project_id}.S_DACPRM.ACO_TAB_ODS`
                    WHERE NOM_TAB_TAMPON = '{NOM_TAB_TAMPON}'
                    LIMIT 1
                """
                df_nom_court_pdk = client.query(query_nom_court_pdk).to_dataframe()
                NOM_COURT = df_nom_court_pdk.iloc[0]['TYP_STRUCTURE_FIC'] if not df_nom_court_pdk.empty else TYP_TAB_ODS
            except Exception as e:
                logger.warning(f"Error getting NOM_COURT for PDK: {{str(e)}}")
                NOM_COURT = TYP_TAB_ODS
            
            # Build column lists
            try:
                # All columns
                query_all_cols_pdk = f"""
                    SELECT NOM_CHAMP
                    FROM `{project_id}.WORK.STRUCTURE_TABLE`
                    WHERE TYP_STRUCTURE_FIC = '{NOM_COURT}'
                    ORDER BY POSITION_CHAMP
                """
                df_all_cols_pdk = client.query(query_all_cols_pdk).to_dataframe()
                ALL_CLE_COMP = ', '.join(df_all_cols_pdk['NOM_CHAMP'].tolist())
                
                # Columns to exclude from comparison
                query_excl_cols_pdk = f"""
                    SELECT NOM_CHAMP
                    FROM `{project_id}.WORK.STRUCTURE_TABLE`
                    WHERE TYP_STRUCTURE_FIC = '{NOM_COURT}'
                        AND FLG_EXCLUSION_DIFF = 'O'
                    ORDER BY POSITION_CHAMP
                """
                df_excl_cols_pdk = client.query(query_excl_cols_pdk).to_dataframe()
                ALL_CLE_AMAJ = ', '.join(df_excl_cols_pdk['NOM_CHAMP'].tolist())
                    
                # Key columns
                query_key_cols_pdk = f"""
                    SELECT NOM_CHAMP
                    FROM `{project_id}.WORK.STRUCTURE_TABLE`
                    WHERE TYP_STRUCTURE_FIC = '{NOM_COURT}'
                        AND FLG_CLE = 'O'
                    ORDER BY POSITION_CHAMP
                """
                df_key_cols_pdk = client.query(query_key_cols_pdk).to_dataframe()
                ALL_CLE = ', '.join(df_key_cols_pdk['NOM_CHAMP'].tolist())
                ALL_CLE_s = ' '.join(df_key_cols_pdk['NOM_CHAMP'].tolist())
            except Exception as e:
                logger.error(f"Error building PDK column lists: {{str(e)}}")
                raise
                
            logger.info(f"PDK exclusion columns length: {len(ALL_CLE_AMAJ)}")
            
            # Count new and old keys
            try:
                query_count_new_keys_pdk = f"""
                    SELECT COUNT(*) as NB_NEW_CLE
                    FROM `{project_id}.WORK._NOUVELLES_CLES{i}`
                """
                df_count_new_keys_pdk = client.query(query_count_new_keys_pdk).to_dataframe()
                NB_NEW_CLE = int(df_count_new_keys_pdk.iloc[0]['NB_NEW_CLE'])
            except Exception as e:
                logger.error(f"Error counting new PDK keys: {{str(e)}}")
                raise
            
            try:
                query_count_old_keys_pdk = f"""
                    SELECT COUNT(*) as NB_OLD_CLE
                    FROM `{project_id}.WORK._OLD_CLE_TO_INSERT{i}`
                """
                df_count_old_keys_pdk = client.query(query_count_old_keys_pdk).to_dataframe()
                NB_OLD_CLE = int(df_count_old_keys_pdk.iloc[0]['NB_OLD_CLE'])
            except Exception as e:
                logger.error(f"Error counting old PDK keys: {{str(e)}}")
                raise
            
            # Log results
            today_dttm = dt.datetime.now()
            try:
                util_dec_aco_alim_suivi_trt(
                    ID_TRT=MV_ID_TRT,
                    DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                    NOM_TRT="CHARGEMENT_ODS",
                    DAT_REEL_TRT=today_dttm,
                    MESG=f"La table {{NOM_TAB_TAMPON}} est chargé dans ODS : {{NB_NEW_CLE}} NOUVELLE(S) CLE(S) et {{NB_OLD_CLE}} MODIFICATION(S) CLES EXISTANTES.",
                    CD_RETOUR=0
                )
            except Exception as e:
                logger.error(f"Error logging PDK results: {{str(e)}}")
            
            # Get MV_EXTENSION from params
            MV_EXTENSION = params.get("MV_EXTENSION", "")
            
            # Move file to archives
            try:
                source_path = f"{REF_FIC_ENT}/ok/{NOM_FIC}"
                dest_path = f"{REF_FIC_ENT}/archives/{NOM_FIC}.{MV_EXTENSION}"
                
                if source_path.startswith("gs://"):
                    from google.cloud import storage
                    
                    source_parts = source_path.replace("gs://", "").split("/", 1)
                    dest_parts = dest_path.replace("gs://", "").split("/", 1)
                    
                    source_bucket = source_parts[0]
                    source_blob = source_parts[1]
                    dest_bucket = dest_parts[0]
                    dest_blob = dest_parts[1]
                    
                    storage_client_gcs = storage.Client(project=project_id)
                    source_bucket_obj = storage_client_gcs.bucket(source_bucket)
                    source_blob_obj = source_bucket_obj.blob(source_blob)
                    dest_bucket_obj = storage_client_gcs.bucket(dest_bucket)
                    
                    source_bucket_obj.copy_blob(source_blob_obj, dest_bucket_obj, dest_blob)
                    source_blob_obj.delete()
                    
                    move_success = True
                else:
                    import shutil
                    shutil.move(source_path, dest_path)
                    move_success = True
            except Exception as e:
                logger.warning(f"Error moving PDK file {NOM_FIC}: {str(e)}")
                move_success = False
            
            logger.info(f"***info move_success: {move_success}")
                
            if move_success:
                today_dttm = dt.datetime.now()
                try:
                    util_dec_aco_alim_suivi_trt(
                        ID_TRT=MV_ID_TRT,
                        DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                        NOM_TRT="CHARGEMENT_ODS",
                        DAT_REEL_TRT=today_dttm,
                        MESG=f"Le fichier {NOM_FIC} a été déplacé dans le répertoire {REF_FIC_ENT}/archives.",
                        CD_RETOUR=0
                    )
                except Exception as e:
                    logger.error(f"Error logging PDK file move: {str(e)}")
                
                # Zip the file
                try:
                    archive_file = f"{NOM_FIC}.{MV_EXTENSION}"
                    archive_path = f"{REF_FIC_ENT}/archives/{archive_file}"
                    zip_path = f"{REF_FIC_ENT}/archives/Archives_{MV_EXTENSION}.zip"
                    
                    if archive_path.startswith("gs://"):
                        import tempfile
                        import zipfile
                        
                        with tempfile.TemporaryDirectory() as tmpdir:
                            archive_parts = archive_path.replace("gs://", "").split("/", 1)
                            archive_bucket = archive_parts[0]
                            archive_blob = archive_parts[1]
                            
                            local_file = f"{tmpdir}/{archive_file}"
                            storage_client_gcs = storage.Client(project=project_id)
                            bucket_obj = storage_client_gcs.bucket(archive_bucket)
                            blob_obj = bucket_obj.blob(archive_blob)
                            blob_obj.download_to_filename(local_file)
                            
                            local_zip = f"{tmpdir}/Archives_{MV_EXTENSION}.zip"
                            
                            zip_parts = zip_path.replace("gs://", "").split("/", 1)
                            zip_blob = zip_parts[1]
                            zip_blob_obj = bucket_obj.blob(zip_blob)
                            
                            if zip_blob_obj.exists():
                                zip_blob_obj.download_to_filename(local_zip)
                                mode = 'a'
                            else:
                                mode = 'w'
                            
                            with zipfile.ZipFile(local_zip, mode) as zf:
                                zf.write(local_file, archive_file)
                            
                            zip_blob_obj.upload_from_filename(local_zip)
                            
                            zip_success = True
                    else:
                        import os
                        import zipfile
                        
                        mode = 'a' if os.path.exists(zip_path) else 'w'
                        with zipfile.ZipFile(zip_path, mode) as zf:
                            zf.write(archive_path, archive_file)
                        
                        zip_success = True
                except Exception as e:
                    logger.warning(f"Error zipping PDK file {NOM_FIC}: {str(e)}")
                    zip_success = False
                    
                    if zip_success:
                        today_dttm = dt.datetime.now()
                        try:
                            util_dec_aco_alim_suivi_trt(
                                ID_TRT=MV_ID_TRT,
                                DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                                NOM_TRT="CHARGEMENT_ODS",
                                DAT_REEL_TRT=today_dttm,
                                MESG=f"Le fichier {NOM_FIC} a été ajouté à l'archive Archives_{MV_EXTENSION}.zip.",
                                CD_RETOUR=0
                            )
                        except Exception as e:
                            logger.error(f"Error logging PDK zip success: {str(e)}")
                        
                        # Remove the unzipped file
                        try:
                            if archive_path.startswith("gs://"):
                                archive_parts = archive_path.replace("gs://", "").split("/", 1)
                                archive_bucket = archive_parts[0]
                                archive_blob = archive_parts[1]
                                
                                storage_client_gcs = storage.Client(project=project_id)
                                bucket_obj = storage_client_gcs.bucket(archive_bucket)
                                blob_obj = bucket_obj.blob(archive_blob)
                                blob_obj.delete()
                            else:
                                import os
                                os.remove(archive_path)
                        except Exception as e:
                            logger.warning(f"Error removing archived PDK file: {str(e)}")
                    else:
                        today_dttm = dt.datetime.now()
                        try:
                            util_dec_aco_alim_suivi_trt(
                                ID_TRT=MV_ID_TRT,
                                DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                                NOM_TRT="CHARGEMENT_ODS",
                                DAT_REEL_TRT=today_dttm,
                                MESG=f"Impossible d'ajouter le fichier {NOM_FIC} à l'archive Archives_{MV_EXTENSION}.zip.",
                                CD_RETOUR=99
                            )
                        except Exception as e:
                            logger.error(f"Error logging PDK zip failure: {str(e)}")
                    
                    # Update tracking table
                    try:
                        query_update_tracking_pdk = f"""
                            UPDATE `{project_id}.{lib_dec_aco_suivi}.ACO_SUIVI_CHG_TAMPON_ODS`
                            SET FLG_TRAITE_ODS = 'O',
                                CD_STATUT_ODS = 'C',
                                DAT_REEL_TRT_ODS = TIMESTAMP('{today_dttm}')
                            WHERE ID_TRT = {MV_ID_TRT}
                                AND DAT_BATCH_TRT = {MV_DAT_BATCH_TRT}
                                AND ID_FIC = '{ID_FIC}'
                        """
                        client.query(query_update_tracking_pdk)
                    except Exception as e:
                        logger.error(f"Error updating tracking table for PDK: {str(e)}")
                        raise
                else:
                    today_dttm = dt.datetime.now()
                    try:
                        util_dec_aco_alim_suivi_trt(
                            ID_TRT=MV_ID_TRT,
                            DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                            NOM_TRT="CHARGEMENT_ODS",
                            DAT_REEL_TRT=today_dttm,
                            MESG=f"Le fichier {NOM_FIC} n'a pas pu être déplacé dans le répertoire {REF_FIC_ENT}/archives.",
                            CD_RETOUR=0
                        )
                    except Exception as e:
                        logger.error(f"Error logging PDK move failure: {str(e)}")
        
            else:
            # Handle validation errors for PDK_SIEG tables
            
            # Count different error types
                try:
                    query_count_err_date_pdk = f"""
                        SELECT COUNT(*) as NB_ERR_DATE
                        FROM `{project_id}.WORK._ERR_DATE{i}`
                    """
                    df_count_err_date_pdk = client.query(query_count_err_date_pdk).to_dataframe()
                    NB_ERR_DATE = int(df_count_err_date_pdk.iloc[0]['NB_ERR_DATE'])
                except Exception as e:
                    logger.error(f"Error counting PDK date errors: {str(e)}")
                    NB_ERR_DATE = 0
            
                try:
                    query_count_err_char_pdk = f"""
                        SELECT COUNT(*) as NB_ERR_CHAR
                        FROM `{project_id}.WORK._ERR_CHAR{i}`
                    """
                    df_count_err_char_pdk = client.query(query_count_err_char_pdk).to_dataframe()
                    NB_ERR_CHAR = int(df_count_err_char_pdk.iloc[0]['NB_ERR_CHAR'])
                except Exception as e:
                    logger.error(f"Error counting PDK char errors: {str(e)}")
                    NB_ERR_CHAR = 0
            
                try:
                    query_count_err_num_pdk = f"""
                        SELECT COUNT(*) as NB_ERR_NUM
                        FROM `{project_id}.WORK._ERR_NUM{i}`
                    """
                    df_count_err_num_pdk = client.query(query_count_err_num_pdk).to_dataframe()
                    NB_ERR_NUM = int(df_count_err_num_pdk.iloc[0]['NB_ERR_NUM'])
                except Exception as e:
                    logger.error(f"Error counting PDK num errors: {str(e)}")
                    NB_ERR_NUM = 0
            
                try:
                    query_count_err_cle_pdk = f"""
                        SELECT COUNT(*) as NB_ERR_CLE
                        FROM `{project_id}.WORK._ERR_CLE{i}`
                    """
                    df_count_err_cle_pdk = client.query(query_count_err_cle_pdk).to_dataframe()
                    NB_ERR_CLE = int(df_count_err_cle_pdk.iloc[0]['NB_ERR_CLE'])
                except Exception as e:
                    logger.error(f"Error counting PDK key errors: {str(e)}")
                    NB_ERR_CLE = 0
            
                # Log error summary
                today_dttm = dt.datetime.now()
                try:
                    util_dec_aco_alim_suivi_trt(
                        ID_TRT=MV_ID_TRT,
                        DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                        NOM_TRT="CHARGEMENT_ODS",
                        DAT_REEL_TRT=today_dttm,
                        MESG=f"La table {NOM_TAB_TAMPON} contient des erreurs : {NB_ERR_DATE} Date(s), {NB_ERR_CHAR} Char(s), {NB_ERR_NUM} Num(s), {NB_ERR_CLE} Clé(s) : Pas de chargement.",
                        CD_RETOUR=99
                    )
                except Exception as e:
                    logger.error(f"Error logging PDK error summary: {str(e)}")
            
                # Move file to KO directory
                try:
                    source_path = f"{REF_FIC_ENT}/ok/{NOM_FIC}"
                    dest_path = f"{REF_FIC_KO}/{NOM_FIC}.{MV_EXTENSION}"
                    
                    if source_path.startswith("gs://"):
                        from google.cloud import storage
                        
                        source_parts = source_path.replace("gs://", "").split("/", 1)
                        dest_parts = dest_path.replace("gs://", "").split("/", 1)
                        
                        source_bucket = source_parts[0]
                        source_blob = source_parts[1]
                        dest_bucket = dest_parts[0]
                        dest_blob = dest_parts[1]
                        
                        storage_client_gcs = storage.Client(project=project_id)
                        source_bucket_obj = storage_client_gcs.bucket(source_bucket)
                        source_blob_obj = source_bucket_obj.blob(source_blob)
                        dest_bucket_obj = storage_client_gcs.bucket(dest_bucket)
                        
                        source_bucket_obj.copy_blob(source_blob_obj, dest_bucket_obj, dest_blob)
                        source_blob_obj.delete()
                        
                        move_ko_success = True
                    else:
                        import shutil
                        shutil.move(source_path, dest_path)
                        move_ko_success = True
                except Exception as e:
                    logger.warning(f"Error moving PDK file {NOM_FIC} to KO: {str(e)}")
                    move_ko_success = False
            
                today_dttm = dt.datetime.now()
                
                if move_ko_success:
                    try:
                        util_dec_aco_alim_suivi_trt(
                            ID_TRT=MV_ID_TRT,
                            DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                            NOM_TRT="CHARGEMENT_ODS",
                            DAT_REEL_TRT=today_dttm,
                            MESG=f"Le fichier {NOM_FIC} a été déplacé dans le répertoire {REF_FIC_KO}.",
                            CD_RETOUR=99
                        )
                    except Exception as e:
                        logger.error(f"Error logging PDK KO move success: {str(e)}")
                else:
                    try:
                        util_dec_aco_alim_suivi_trt(
                            ID_TRT=MV_ID_TRT,
                            DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                            NOM_TRT="CHARGEMENT_ODS",
                            DAT_REEL_TRT=today_dttm,
                            MESG=f"Le fichier {NOM_FIC} n'a pas pu être déplacé dans le répertoire {REF_FIC_KO}.",
                            CD_RETOUR=99
                        )
                    except Exception as e:
                        logger.error(f"Error logging PDK KO move failure: {str(e)}")
            
                # Update tracking table with error status
                try:
                    query_update_error_pdk = f"""
                        UPDATE `{project_id}.{lib_dec_aco_suivi}.ACO_SUIVI_CHG_TAMPON_ODS`
                        SET CD_STATUT_ODS = 'E',
                            DAT_REEL_TRT_ODS = TIMESTAMP('{today_dttm}')
                        WHERE ID_TRT = {MV_ID_TRT}
                            AND DAT_BATCH_TRT = {MV_DAT_BATCH_TRT}
                            AND ID_FIC = '{ID_FIC}'
                    """
                    client.query(query_update_error_pdk)
                except Exception as e:
                    logger.error(f"Error updating tracking table for PDK errors: {str(e)}")
                    raise

                else:
                    # Handle tables with no keys (not PDK_SIEG)
                    today_dttm = dt.datetime.now()
                        
                        # Move file to KO directory
                    try:
                        source_path = f"{REF_FIC_ENT}/ok/{NOM_FIC}"
                        dest_path = f"{REF_FIC_KO}/{NOM_FIC}...{MV_EXTENSION}"
                            
                        if source_path.startswith("gs://"):
                            from google.cloud import storage
                                
                            source_parts = source_path.replace("gs://", "").split("/", 1)
                            dest_parts = dest_path.replace("gs://", "").split("/", 1)
                            
                            source_bucket = source_parts[0]
                            source_blob = source_parts[1]
                            dest_bucket = dest_parts[0]
                            dest_blob = dest_parts[1]
                            
                            storage_client_gcs = storage.Client(project=project_id)
                            source_bucket_obj = storage_client_gcs.bucket(source_bucket)
                            source_blob_obj = source_bucket_obj.blob(source_blob)
                            dest_bucket_obj = storage_client_gcs.bucket(dest_bucket)
                            
                            source_bucket_obj.copy_blob(source_blob_obj, dest_bucket_obj, dest_blob)
                            source_blob_obj.delete()
                            
                            move_no_key_success = True
                        else:
                            import shutil
                            shutil.move(source_path, dest_path)
                            move_no_key_success = True
                    except Exception as e:
                        logger.warning(f"Error moving file {NOM_FIC} (no keys): {str(e)}")
                        move_no_key_success = False
                        
                    if move_no_key_success:
                        try:
                            util_dec_aco_alim_suivi_trt(
                                ID_TRT=MV_ID_TRT,
                                DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                                NOM_TRT="CHARGEMENT_ODS",
                                DAT_REEL_TRT=today_dttm,
                                MESG=f"Le fichier {NOM_FIC} a été déplacé dans le répertoire {REF_FIC_KO}.",
                                CD_RETOUR=99
                            )
                        except Exception as e:
                            logger.error(f"Error logging no-key file move success: {str(e)}")
                    else:
                        try:
                            util_dec_aco_alim_suivi_trt(
                                ID_TRT=MV_ID_TRT,
                                DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                                NOM_TRT="CHARGEMENT_ODS",
                                DAT_REEL_TRT=today_dttm,
                                MESG=f"Le fichier {NOM_FIC} n'a pas pu être déplacé dans le répertoire {REF_FIC_KO}.",
                                CD_RETOUR=99
                            )
                        except Exception as e:
                            logger.error(f"Error logging no-key file move failure: {str(e)}")
                    
                        try:
                            util_dec_aco_alim_suivi_trt(
                                ID_TRT=MV_ID_TRT,
                                DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                                NOM_TRT="CHARGEMENT_ODS",
                                DAT_REEL_TRT=today_dttm,
                                MESG=f"Pas de clé définie sur la table {NOM_TAB_TAMPON} : Pas de chargement.",
                                CD_RETOUR=99
                            )
                        except Exception as e:
                            logger.error(f"Error logging no-key message: {str(e)}")
                        
                        # Update tracking table with invalid status
                        try:
                            query_update_invalid = f"""
                                UPDATE `{project_id}.{lib_dec_aco_suivi}.ACO_SUIVI_CHG_TAMPON_ODS`
                                SET CD_STATUT_ODS = 'I',
                                    DAT_REEL_TRT_ODS = TIMESTAMP('{today_dttm}')
                                WHERE ID_TRT = {MV_ID_TRT}
                                    AND DAT_BATCH_TRT = {MV_DAT_BATCH_TRT}
                                    AND ID_FIC = '{ID_FIC}'
                            """
                            client.query(query_update_invalid)
                        except Exception as e:
                            logger.error(f"Error updating tracking table for no-key table: {str(e)}")
                            raise

    else:
        # No tables to load into ODS
        today_dttm = dt.datetime.now()
        try:
            util_dec_aco_alim_suivi_trt(
                ID_TRT=MV_ID_TRT,
                DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                NOM_TRT="CHARGEMENT_ODS",
                DAT_REEL_TRT=today_dttm,
                MESG="Aucune table à charger dans la partie ODS.",
                CD_RETOUR=0
            )
        except Exception as e:
            logger.error(f"Error logging no tables message: {str(e)}")

    # Get max ID_TRT from ACO_SUIVI_BATCH for rejection analysis
    try:
        query_max_id_trt = f"""
            SELECT MAX(ID_TRT) as ID_TRT_SE_MAX
            FROM `{project_id}.S_ACOSUI.ACO_SUIVI_BATCH`
        """
        df_max_id_trt = client.query(query_max_id_trt).to_dataframe()
        ID_TRT_SE_MAX = int(df_max_id_trt.iloc[0]['ID_TRT_SE_MAX'])
    except Exception as e:
        logger.error(f"Error getting max ID_TRT: {str(e)}")
        raise

    # Get rejection records
    try:
        query_rejet_se = f"""
            CREATE OR REPLACE TABLE `{project_id}.WORK.REJET_SE` AS
            SELECT DISTINCT *
            FROM `{project_id}.S_ACOSUI.ACO_SUIVI_FIC`
            WHERE ID_TRT = {ID_TRT_SE_MAX}
                AND NB_REJET > 0
        """
        client.query(query_rejet_se)
    except Exception as e:
        logger.error(f"Error creating REJET_SE table: {str(e)}")
        raise

    # Count rejections
    try:
        query_count_rejet = f"""
            SELECT COUNT(ID_TRT) as NB_REJET
            FROM `{project_id}.WORK.REJET_SE`
        """
        df_count_rejet = client.query(query_count_rejet).to_dataframe()
        NB_REJET = int(df_count_rejet.iloc[0]['NB_REJET'])
    except Exception as e:
        logger.error(f"Error counting rejections: {str(e)}")
        raise

    # Get FLG_DBL from params
    FLG_DBL = params.get("FLG_DBL", "N")

    # Handle different combinations of duplicates and rejections
    if FLG_DBL == "O" and NB_REJET == 0:
        # Duplicates but no rejections
        intro_rej = "<b>Pas de rejets dans espace échange</b><br><br>"
        intro_dbl = "<b>Liste de(s) tables(s) du Tampon avec doublons:</b><br>"

        try:
            query_dbl_text = f"""
                SELECT CONCAT(
                    'Table ', TABLE, ' : ',
                    CAST(NB_DOUBLONS AS STRING), ' enregistrements sur ',
                    CAST(NB_OBS_TABLE AS STRING), ' au total soit ',
                    CAST(ROUND(NB_DOUBLONS * 100.0 / NB_OBS_TABLE, 2) AS STRING), ' %'
                ) as text_line
                FROM `{project_id}.S_DACSUI.ACO_SUIVI_DOUBLONS`
                WHERE ID_TRT = {MV_ID_TRT}
            """
            df_dbl_text = client.query(query_dbl_text).to_dataframe()
            l_txt_libre_DBL = '<br>'.join(df_dbl_text['text_line'].tolist())
        except Exception as e:
            logger.error(f"Error building duplicate text: {str(e)}")
            l_txt_libre_DBL = ""

        l_txt_libre = intro_rej + intro_dbl + l_txt_libre_DBL
        logger.info(l_txt_libre)

        try:
            util_dec_aco_mail_envoi(
                P_ID_GROUPE="MAIL_DAC_DOUBLON_TPN",
                P_DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                P_ID_TRT=MV_ID_TRT,
                P_TYPE_MAIL="TXT_LIBRE",
                P_TXT_LIBRE=l_txt_libre,
            )
        except Exception as e:
            logger.error(f"Error sending duplicate email: {str(e)}")

    elif FLG_DBL == "O" and NB_REJET > 0:
        # Both duplicates and rejections
        intro_dbl = "<b><br><br>Liste de(s) tables(s) du Tampon avec doublons:</b><br>"

        try:
            query_dbl_text = f"""
                SELECT CONCAT(
                    'Table ', TABLE, ' : ',
                    CAST(NB_DOUBLONS AS STRING), ' enregistrements sur ',
                    CAST(NB_OBS_TABLE AS STRING), ' au total soit ',
                    CAST(ROUND(NB_DOUBLONS * 100.0 / NB_OBS_TABLE, 2) AS STRING), ' %'
                ) as text_line
                FROM `{project_id}.S_DACSUI.ACO_SUIVI_DOUBLONS`
                WHERE ID_TRT = {MV_ID_TRT}
            """
            df_dbl_text = client.query(query_dbl_text).to_dataframe()
            l_txt_libre_DBL = '<br>'.join(df_dbl_text['text_line'].tolist())
        except Exception as e:
            logger.error(f"Error building duplicate text: {str(e)}")
            l_txt_libre_DBL = ""

        intro_rej = "<b>Liste de(s) tables(s) de espace échange ayant des rejets:</b><br>"

        try:
            query_rej_text = f"""
                SELECT CONCAT(
                    'Table ', ID_FIC_A_CTRL, ' : ',
                    CAST(NB_REJET AS STRING), ' enregistrements sur ',
                    CAST(NB_ENREG AS STRING), ' au total soit ',
                    CAST(ROUND(NB_REJET * 100.0 / NB_ENREG, 2) AS STRING), ' %'
                ) as text_line
                FROM `{project_id}.WORK.REJET_SE`
            """
            df_rej_text = client.query(query_rej_text).to_dataframe()
            l_txt_libre_REJ = '<br>'.join(df_rej_text['text_line'].tolist())
        except Exception as e:
            logger.error(f"Error building rejection text: {str(e)}")
            l_txt_libre_REJ = ""

        l_txt_libre = intro_rej + l_txt_libre_REJ + intro_dbl + l_txt_libre_DBL
        logger.info(l_txt_libre)

        try:
            util_dec_aco_mail_envoi(
                P_ID_GROUPE="MAIL_DAC_DOUBLON_TPN",
                P_DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                P_ID_TRT=MV_ID_TRT,
                P_TYPE_MAIL="TXT_LIBRE",
                P_TXT_LIBRE=l_txt_libre
            )
        except Exception as e:
            logger.error(f"Error sending duplicate/rejection email: {str(e)}")

    elif FLG_DBL == "N" and NB_REJET > 0:
        # Rejections but no duplicates
        intro_dbl = "<b><br><br>Pas de Doublons dans les tables Tampons</b><br>"
        intro_rej = "<b>Liste de(s) tables(s) de espace échange ayant des rejets:</b><br>"
        
        try:
            query_rej_text = f"""
                SELECT CONCAT(
                    'Table ', ID_FIC_A_CTRL, ' : ',
                    CAST(NB_REJET AS STRING), ' enregistrements sur ',
                    CAST(NB_ENREG AS STRING), ' au total soit ',
                    CAST(ROUND(NB_REJET * 100.0 / NB_ENREG, 2) AS STRING), ' %'
                ) as text_line
                FROM `{project_id}.WORK.REJET_SE`
            """
            df_rej_text = client.query(query_rej_text).to_dataframe()
            l_txt_libre_REJ = '<br>'.join(df_rej_text['text_line'].tolist())
        except Exception as e:
            logger.error(f"Error building rejection text: {str(e)}")
            l_txt_libre_REJ = ""
        
        l_txt_libre = intro_rej + l_txt_libre_REJ + intro_dbl
        logger.info(l_txt_libre)
        
        try:
            util_dec_aco_mail_envoi(
                P_ID_GROUPE="MAIL_DAC_DOUBLON_TPN",
                P_DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                P_ID_TRT=MV_ID_TRT,
                P_TYPE_MAIL="TXT_LIBRE",
                P_TXT_LIBRE=l_txt_libre
            )
        except Exception as e:
            logger.error(f"Error sending rejection email: {str(e)}")

    # Finalize treatment tracking
    try:
        UTIL_DEC_ACO_ALIM_SUIVI_TRT_FIN(
            MV_ID_TRT=MV_ID_TRT,
            DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
            NOM_TRT="CHARGEMENT_ODS",
            MSG="Début Traitement CHARGEMENT_ODS",
        )
    except Exception as e:
        logger.error(f"Error finalizing treatment tracking: {str(e)}")

    # Get NB_ERR_TRT from params (should be set by previous processing)
    NB_ERR_TRT = params.get("NB_ERR_TRT", 0)

    # Log final status
    today_dttm = dt.datetime.now()
    if NB_ERR_TRT == 0:
        try:
            util_dec_aco_alim_suivi_trt(
                ID_TRT=MV_ID_TRT,
                DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                NOM_TRT="CHARGEMENT_ODS",
                DAT_REEL_TRT=today_dttm,
                MESG="Fin du traitement CHARGEMENT_ODS OK",
                CD_RETOUR=0
            )
        except Exception as e:
            logger.error(f"Error logging final OK status: {str(e)}")
    else:
        try:
            util_dec_aco_alim_suivi_trt(
                ID_TRT=MV_ID_TRT,
                DAT_BATCH_TRT=MV_DAT_BATCH_TRT,
                NOM_TRT="CHARGEMENT_ODS",
                DAT_REEL_TRT=today_dttm,
                MESG="Fin du traitement CHARGEMENT_ODS KO",
                CD_RETOUR=99
            )
        except Exception as e:
            logger.error(f"Error logging final KO status: {str(e)}")

    logger.info("CHARGEMENT_ODS process completed")