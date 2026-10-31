def get_severity(confidence):

    if confidence >= 0.90:
        return "🔴 High"

    elif confidence >= 0.70:
        return "🟡 Moderate"

    return "🟢 Low"


def models_agree(
        class1,
        class2):

    return class1 == class2