import pandas as pd
import matplotlib.pyplot as plt


def plot_feature_importance(model, feature_names):
    weights = model.linear.weight.data.numpy().flatten()
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': weights})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    print(feature_importance)
    plt.figure(fsize=(10, 6))
    plt.bar(feature_importance['Feature'], feature_importance['Importance'])
    plt.xticks(rotation=45)
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()