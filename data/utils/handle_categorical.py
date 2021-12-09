import pandas as pd

time_dict = {'nan': -1, '7AM': 0, '10AM': 1, '2PM': 2, '6PM': 3, '10PM': 4}
expiration_dict = {'nan': -1, '2h': 0, '1d': 1}
age_dict = {'nan': -1, 'below21': 0, '21': 1, '26': 2, '31': 3, '36': 4, '41': 5, '46': 6, '50plus': 7}
education_dict = {'nan': -1, 'Some High School': 0, 'High School Graduate': 1, 'Associates degree': 2,
                  'Some college - no degree': 3, 'Bachelors degree': 4, 'Graduate degree (Masters or Doctorate)': 5}
income_dict = {'nan': -1, 'Less than $12500': 0, '$12500 - $24999': 1, '$25000 - $37499': 2, '$37500 - $49999': 3,
               '$50000 - $62499': 4, '$62500 - $74999': 5, '$75000 - $87499': 6, '$87500 - $99999': 7,
               '$100000 or More': 8}
bars_dict = {'nan': -1, 'never': 0, 'less1': 1, r'1~3': 2, r'4~8': 3, 'gt8': 4}
ordinal_dict = {'time': time_dict,
                'expiration': expiration_dict,
                'age': age_dict,
                'education': education_dict,
                'income': income_dict,
                'Bar': bars_dict,
                'CoffeeHouse': bars_dict,
                'CarryAway': bars_dict,
                'RestaurantLessThan20': bars_dict,
                'Restaurant20To50': bars_dict}


def make_ordinal(df, categorical_variables):
    def make_ordinal_row(row, variable):
        row[variable] = ordinal_dict[variable][row[variable]]
        return row

    for variable in categorical_variables:
        # handle toCoupon differently
        if 'toCoupon' in variable:
            df = merge_tocoupon_to_ordinal(df)
        else:
            df = df.apply(lambda x: make_ordinal_row(x, variable), axis=1)

    return df


def merge_tocoupon_to_ordinal(df):
    def label_coupon_time(row):
        if not row['toCoupon_GEQ5min']:
            return 0
        if row['toCoupon_GEQ5min'] and not row['toCoupon_GEQ15min']:
            return 1
        if row['toCoupon_GEQ15min'] and not row['toCoupon_GEQ25min']:
            return 2
        if row['toCoupon_GEQ25min']:
            return 3
        else:
            return None

    df['coupon_time'] = df.apply(label_coupon_time, axis=1)
    df.drop(columns='toCoupon_GEQ5min', inplace=True)
    df.drop(columns='toCoupon_GEQ15min', inplace=True)
    df.drop(columns='toCoupon_GEQ25min', inplace=True)

    return df


def make_one_hot(df, categorical_variables):
    for variable in categorical_variables:
        discarded = pd.get_dummies(df[variable], prefix=variable)
        if discarded.shape[1] == 2:
            discarded = discarded.iloc[:, 0]
        df = pd.concat([df, discarded], axis=1)
        df.drop([variable], axis=1, inplace=True)

    return df
