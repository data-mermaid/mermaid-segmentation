from matplotlib import pyplot as plt

def get_legend_elements(annotations):
    unique_benthic = annotations[['benthic_attribute_name', 'benthic_color']].drop_duplicates()
    benthic_legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=color, label=name, markersize=10)
                    for name, color in zip(unique_benthic['benthic_attribute_name'], 
                                        unique_benthic['benthic_color'])]

    unique_growth = annotations[['growth_form_name', 'growth_form_marker']].astype(str).drop_duplicates()

    growth_legend_elements = [plt.Line2D([0], [0], marker=marker, color='#4c72b0', 
                                    label=name, markersize=10, linestyle='None')
                                for name, marker in zip(unique_growth['growth_form_name'], 
                                                unique_growth['growth_form_marker'])]

    return benthic_legend_elements, growth_legend_elements