import numpy as np
import pandas as pd
import os


def load_files(directory='../input'):
    """read input files and return a dict with data frames
    """
    # list files in directory
    files = os.listdir(directory)
    output = {}
    for file in files:
        # trim file extension from dict key
        dict_key = file.split('.')[0] 
        
        # assign 
        output[dict_key] = pd.read_csv(os.path.join(directory, file))
    return output['sample_submission'], output['X_train'], output['X_test'], output['y_train']


def normalize_quaternion(df, cols=['orientation_{}'.format(name) for name in ['X', 'Y', 'Z', 'W']]):
    """Normalize orientations to unit quarternion
    """
    
    norm_const_sq = np.sum(df[cols] ** 2, axis=1)
    norm_const = np.sqrt(norm_const_sq)
    for c in cols:
        df[c] = np.divide(df[c].values, norm_const)
    
    return df

def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (np.multiply(w, x) + np.multiply(y, z))
    t1 = +1.0 - 2.0 * (np.multiply(x, x) + np.multiply(y, y))
    X = np.degrees(np.arctan2(t0, t1))

    t2 = +2.0 * (np.multiply(w, y) - np.multiply(z, x))
    t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
    Y = np.degrees(np.arcsin(t2))

    t3 = +2.0 * (np.multiply(w, z) + np.multiply(x, y))
    t4 = +1.0 - 2.0 * (np.multiply(y, y) + np.multiply(z, z))
    Z = np.degrees(np.arctan2(t3, t4))

    return X, Y, Z


def add_euler(df, cols=['orientation_{}'.format(name) for name in ['X', 'Y', 'Z', 'W']]):
    col_x = [c for c in cols if '_X' in c]
    col_y = [c for c in cols if '_Y' in c]
    col_z = [c for c in cols if '_Z' in c]
    col_w = [c for c in cols if '_W' in c]
    
    x = df[col_x].values
    y = df[col_y].values
    z = df[col_z].values
    w = df[col_w].values
    df['euler_X'], df['euler_Y'], df['euler_Z'] = np.nan, np.nan, np.nan

    df['euler_X'], df['euler_Y'], df['euler_Z'] = quaternion_to_euler(x, y, z, w)
    
    return df


def add_total_custom_features(df):
    """Add dot and cross products to the previous line
    """
    # subtract gravity acceleration from linear acceleration Z
    df['linear_acceleration_Z'] += 9.8
    
    # modulo of total angular velocity
    cols_angular = [c for c in df.columns if 'angular' in c]
    for c in cols_angular:
        orient = c.split('_')[-1]
        df['angular_velocity_pow2_{}'.format(orient)] = np.power(df[c], 2)
        
    pow2_angular = [c for c in df.columns if 'angular_velocity_pow2' in c]
    df['total_angular_velocity'] = np.sqrt(np.sum(df[pow2_angular], axis=1))
    
    # modulo of total linear acceleration
    cols_acc = [c for c in df.columns if 'acceleration' in c]
    for c in cols_acc:
        orient = c.split('_')[-1]
        df['linear_acceleration_pow2_{}'.format(orient)] = np.power(df[c], 2)
        
    pow2_acc = [c for c in df.columns if 'linear_acceleration_pow2' in c]
    df['total_linear_acc'] = np.sqrt(np.sum(df[pow2_acc], axis=1))
    
    # ratios
    #df['ratio_acc_vel'] = df['total_linear_acc'] / (df['total_angular_velocity'] + 1e-6)
    
    return df


def transform_ts_3d_array(df, group_col, feat_cols):
    """Transform array of times series and their features to an 3d array
    with shape (no_times_series, time_series_length, no_features)
    """
    ts_len = df.groupby(group_col)[group_col].count().max() # length of ts
    ts_no_feats = df[feat_cols].shape[1] # number of cols
    transformed_df = []
    
    for id_ in df[group_col].unique():
        transformed_df.append(
            df.loc[df[group_col] == id_][feat_cols].values)
    
    return np.array(transformed_df) #.reshape(-1, ts_len, ts_no_feats)


def preprocess_data(df_in, group_col, cum=False):
    """All preprocessing steps together
    """
    df = df_in.copy()
    # normalize quaternions
    df_norm = normalize_quaternion(df)
    # add euler angles
    df_norm = add_euler(df_norm)
    # add total angular velocity
    df_norm = add_total_custom_features(df_norm)    
    # get features cols
    features_cols = [c for c in df_norm.columns if df_norm[c].dtype == np.float64]
    # convert to 3d shape
    df_3d = transform_ts_3d_array(df_norm, group_col, features_cols)

    return df_3d, features_cols, df_norm