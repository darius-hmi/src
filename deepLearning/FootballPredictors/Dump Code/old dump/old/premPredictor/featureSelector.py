import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv('cleaned_file_noNull_no_noNonInteger.csv')

# Initial feature and target selection
features = [
        'Home', 'xG', 'xG.1', 'Away',
        # Home team stats
        'Home_Rk', 'Home_MP', 'Home_W', 'Home_D', 'Home_L', 'Home_GF', 'Home_GA', 'Home_GD', 'Home_Pts', 'Home_Pts/MP',
        'Home_xG', 'Home_xGA', 'Home_xGD', 'Home_xGD/90',
        'Home_# Pl', 'Home_90s',
        'Home_Cmp', 'Home_Att', 'Home_Cmp%', 'Home_TotDist', 'Home_PrgDist',
        'Home_Cmp.1', 'Home_Att.1', 'Home_Cmp%.1',
        'Home_Cmp.2', 'Home_Att.2', 'Home_Cmp%.2',
        'Home_Cmp.3', 'Home_Att.3', 'Home_Cmp%.3',
        'Home_Ast', 'Home_xAG', 'Home_xA', 'Home_A-xAG', 'Home_KP',
        'Home_01-Mar', 'Home_PPA', 'Home_CrsPA', 'Home_PrgP', 'Home_Live', 'Home_Dead',
        'Home_FK', 'Home_TB', 'Home_Sw', 'Home_Crs', 'Home_TI', 'Home_CK', 'Home_In', 'Home_Out', 'Home_Str', 'Home_Off',
        'Home_Blocks', 'Home_SCA', 'Home_SCA90', 'Home_PassLive', 'Home_PassDead', 'Home_TO', 'Home_Sh', 'Home_Fld', 'Home_Def',
        'Home_GCA', 'Home_GCA90',
        'Home_PassLive.1', 'Home_PassDead.1', 'Home_TO.1', 'Home_Sh.1', 'Home_Fld.1', 'Home_Def.1',
        'Home_Tkl', 'Home_TklW', 'Home_Def 3rd', 'Home_Mid 3rd', 'Home_Att 3rd',
        'Home_Tkl.1', 'Home_Tkl%', 'Home_Lost', 'Home_Pass', 'Home_Int', 'Home_Tkl+Int', 'Home_Clr', 'Home_Err', 'Home_Poss',
        'Home_Touches', 'Home_Def Pen', 'Home_Att Pen', 'Home_Succ', 'Home_Succ%', 'Home_Tkld', 'Home_Tkld%',
        'Home_Carries', 'Home_PrgC', 'Home_CPA', 'Home_Mis', 'Home_Dis', 'Home_Rec', 'Home_PrgR',
        'Home_HMP', 'Home_HW', 'Home_HD', 'Home_HL', 'Home_HGF', 'Home_HGA', 'Home_HGD', 'Home_HPts', 'Home_HPts/MP',
        'Home_HxG', 'Home_HxGA', 'Home_HxGD', 'Home_HxGD/90',
        'Home_AMP', 'Home_AW', 'Home_AD', 'Home_AL', 'Home_AGF', 'Home_AGA', 'Home_AGD', 'Home_APts', 'Home_APts/MP',
        'Home_AxG', 'Home_AxGA', 'Home_AxGD', 'Home_AxGD/90',
        'Home_Age', 'Home_Min', 'Home_Mn/MP', 'Home_Min%', 'Home_Starts', 'Home_Mn/Start', 'Home_Subs', 'Home_Mn/Sub',
        'Home_unSub', 'Home_PPM', 'Home_onG', 'Home_onGA', 'Home_+/-', 'Home_+/-90', 'Home_onxG', 'Home_onxGA', 'Home_xG+/-',
        'Home_xG+/-90', 'Home_CrdY', 'Home_CrdR', 'Home_2CrdY', 'Home_Fls', 'Home_PKwon', 'Home_PKcon', 'Home_OG', 'Home_Recov',
        'Home_Won', 'Home_Won%',
        'Home_Gls', 'Home_G+A', 'Home_G-PK', 'Home_PK', 'Home_PKatt', 'Home_npxG', 'Home_npxG+xAG',
        'Home_Gls.1', 'Home_Ast.1', 'Home_G+A.1', 'Home_G-PK.1', 'Home_G+A-PK', 'Home_xG.1', 'Home_xAG.1', 'Home_xG+xAG',
        'Home_npxG.1', 'Home_npxG+xAG.1', 'Home_GA90', 'Home_SoTA', 'Home_Saves', 'Home_Save%', 'Home_CS', 'Home_CS%',
        'Home_PKA', 'Home_PKsv', 'Home_PKm', 'Home_PSxG', 'Home_PSxG/SoT', 'Home_PSxG+/-', 'Home_/90',
        'Home_Att (GK)', 'Home_Thr', 'Home_Launch%', 'Home_AvgLen', 'Home_Launch%.1', 'Home_AvgLen.1',
        'Home_Opp', 'Home_Stp', 'Home_Stp%', 'Home_#OPA', 'Home_#OPA/90', 'Home_SoT', 'Home_SoT%', 'Home_Sh/90',
        'Home_SoT/90', 'Home_G/Sh', 'Home_G/SoT', 'Home_Dist', 'Home_npxG/Sh', 'Home_G-xG', 'Home_np:G-xG',
        # Away team stats
        'Away_Rk', 'Away_MP', 'Away_W', 'Away_D', 'Away_L', 'Away_GF', 'Away_GA', 'Away_GD', 'Away_Pts', 'Away_Pts/MP',
        'Away_xG', 'Away_xGA', 'Away_xGD', 'Away_xGD/90',
        'Away_# Pl', 'Away_90s',
        'Away_Cmp', 'Away_Att', 'Away_Cmp%', 'Away_TotDist', 'Away_PrgDist',
        'Away_Cmp.1', 'Away_Att.1', 'Away_Cmp%.1',
        'Away_Cmp.2', 'Away_Att.2', 'Away_Cmp%.2',
        'Away_Cmp.3', 'Away_Att.3', 'Away_Cmp%.3',
        'Away_Ast', 'Away_xAG', 'Away_xA', 'Away_A-xAG', 'Away_KP',
        'Away_01-Mar', 'Away_PPA', 'Away_CrsPA', 'Away_PrgP', 'Away_Live', 'Away_Dead',
        'Away_FK', 'Away_TB', 'Away_Sw', 'Away_Crs', 'Away_TI', 'Away_CK', 'Away_In', 'Away_Out', 'Away_Str', 'Away_Off',
        'Away_Blocks', 'Away_SCA', 'Away_SCA90', 'Away_PassLive', 'Away_PassDead', 'Away_TO', 'Away_Sh', 'Away_Fld', 'Away_Def',
        'Away_GCA', 'Away_GCA90',
        'Away_PassLive.1', 'Away_PassDead.1', 'Away_TO.1', 'Away_Sh.1', 'Away_Fld.1', 'Away_Def.1',
        'Away_Tkl', 'Away_TklW', 'Away_Def 3rd', 'Away_Mid 3rd', 'Away_Att 3rd',
        'Away_Tkl.1', 'Away_Tkl%', 'Away_Lost', 'Away_Pass', 'Away_Int', 'Away_Tkl+Int', 'Away_Clr', 'Away_Err', 'Away_Poss',
        'Away_Touches', 'Away_Def Pen', 'Away_Att Pen', 'Away_Succ', 'Away_Succ%', 'Away_Tkld', 'Away_Tkld%',
        'Away_Carries', 'Away_PrgC', 'Away_CPA', 'Away_Mis', 'Away_Dis', 'Away_Rec', 'Away_PrgR',
        'Away_HMP', 'Away_HW', 'Away_HD', 'Away_HL', 'Away_HGF', 'Away_HGA', 'Away_HGD', 'Away_HPts', 'Away_HPts/MP',
        'Away_HxG', 'Away_HxGA', 'Away_HxGD', 'Away_HxGD/90',
        'Away_AMP', 'Away_AW', 'Away_AD', 'Away_AL', 'Away_AGF', 'Away_AGA', 'Away_AGD', 'Away_APts', 'Away_APts/MP',
        'Away_AxG', 'Away_AxGA', 'Away_AxGD', 'Away_AxGD/90',
        'Away_Age', 'Away_Min', 'Away_Mn/MP', 'Away_Min%', 'Away_Starts', 'Away_Mn/Start', 'Away_Subs', 'Away_Mn/Sub',
        'Away_unSub', 'Away_PPM', 'Away_onG', 'Away_onGA', 'Away_+/-', 'Away_+/-90', 'Away_onxG', 'Away_onxGA', 'Away_xG+/-',
        'Away_xG+/-90', 'Away_CrdY', 'Away_CrdR', 'Away_2CrdY', 'Away_Fls', 'Away_PKwon', 'Away_PKcon', 'Away_OG', 'Away_Recov',
        'Away_Won', 'Away_Won%',
        'Away_Gls', 'Away_G+A', 'Away_G-PK', 'Away_PK', 'Away_PKatt', 'Away_npxG', 'Away_npxG+xAG',
        'Away_Gls.1', 'Away_Ast.1', 'Away_G+A.1', 'Away_G-PK.1', 'Away_G+A-PK', 'Away_xG.1', 'Away_xAG.1', 'Away_xG+xAG',
        'Away_npxG.1', 'Away_npxG+xAG.1', 'Away_GA90', 'Away_SoTA', 'Away_Saves', 'Away_Save%', 'Away_CS', 'Away_CS%',
        'Away_PKA', 'Away_PKsv', 'Away_PKm', 'Away_PSxG', 'Away_PSxG/SoT', 'Away_PSxG+/-', 'Away_/90',
        'Away_Att (GK)', 'Away_Thr', 'Away_Launch%', 'Away_AvgLen', 'Away_Launch%.1', 'Away_AvgLen.1',
        'Away_Opp', 'Away_Stp', 'Away_Stp%', 'Away_#OPA', 'Away_#OPA/90', 'Away_SoT', 'Away_SoT%', 'Away_Sh/90',
        'Away_SoT/90', 'Away_G/Sh', 'Away_G/SoT', 'Away_Dist', 'Away_npxG/Sh', 'Away_G-xG', 'Away_np:G-xG',
        'Season'
    ]

# Ensure that 'Result' is the target variable
target = 'Result'
X = df[features]
y = df[target]

# Encode the target variable if it is categorical
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Identify categorical columns for encoding
categorical_features = ['Home', 'Away', 'Season']  # List your categorical columns here
numerical_features = [col for col in X.columns if col not in categorical_features]

# Preprocessing pipeline for categorical data
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessing pipeline for numerical data
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Combine preprocessing for both types of data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Define the overall pipeline with SelectKBest for feature selection
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(score_func=chi2, k=20)),  # Default feature selection; will be updated
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Extract selected features
selector_kbest = pipeline.named_steps['feature_selection']
selected_features_kbest = X.columns[selector_kbest.get_support()]
print("Selected Features (KBest):", selected_features_kbest)

# Evaluate the model performance
y_pred_kbest = pipeline.predict(X_test)
print("\nModel Evaluation (KBest Features):")
print("Accuracy:", accuracy_score(y_test, y_pred_kbest))
print(classification_report(y_test, y_pred_kbest, target_names=label_encoder.classes_))

# Cross-validation on the entire pipeline
cv_scores_kbest = cross_val_score(pipeline, X_train, y_train, cv=5)
print("\nCross-Validation Scores (KBest Features):", cv_scores_kbest)
print("Mean CV Score (KBest Features):", cv_scores_kbest.mean())

# Feature Selection using SelectFromModel
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(preprocessor.transform(X_train), y_train)

selector_model = SelectFromModel(model, threshold='mean', prefit=True)
X_train_model = selector_model.transform(preprocessor.transform(X_train))
X_test_model = selector_model.transform(preprocessor.transform(X_test))

# Train a RandomForest model on the selected features from SelectFromModel
model_model = RandomForestClassifier(n_estimators=100, random_state=42)
model_model.fit(X_train_model, y_train)
y_pred_model = model_model.predict(X_test_model)

# Evaluate the model performance using classification report and accuracy
print("\nModel Evaluation (SelectFromModel Features):")
print("Accuracy:", accuracy_score(y_test, y_pred_model))
print(classification_report(y_test, y_pred_model, target_names=label_encoder.classes_))

# Cross-validation on selected features from SelectFromModel
cv_scores_model = cross_val_score(model_model, X_train_model, y_train, cv=5)
print("\nCross-Validation Scores (SelectFromModel Features):", cv_scores_model)
print("Mean CV Score (SelectFromModel Features):", cv_scores_model.mean())
