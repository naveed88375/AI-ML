#Function to perform feature extraction
def feature_extraction(df, dtype):
    #Create frame to store features
    df_feat = df.iloc[:,:3]
    #Calculate mean, variance, max and minimum values
    df_feat[dtype+'_mean'] = df.mean(axis=1, numeric_only=True)
    df_feat[dtype+'_var'] = df.var(axis=1, numeric_only=True)
    df_feat[dtype+'_max'] = df.max(axis=1, numeric_only=True)
    df_feat[dtype+'_min'] = df.min(axis=1, numeric_only=True)
    df_feat=df_feat.reset_index().drop('index',axis=1)
    return df_feat