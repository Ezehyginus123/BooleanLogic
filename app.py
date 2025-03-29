import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import symbols, SOPform
from itertools import product

st.title("Karnaugh Map Simplifier")

# Get user input for binary output
binary_output = st.text_input("Enter Binary Output (e.g., 11011000, 0110)", "")

# Add a Submit button
if st.button("Submit"):
    if not binary_output or not all(c in "01" for c in binary_output):
        st.error("Invalid input! Enter a binary string containing only 0s and 1s.")
    else:
        num_inputs = (len(binary_output) - 1).bit_length()
        minterms = [i for i, bit in enumerate(binary_output[::-1]) if bit == '1']
        variables = [symbols(chr(65 + i)) for i in range(num_inputs)]
        
        if minterms:
            simplified_expr = SOPform(variables, minterms)
        else:
            simplified_expr = "0"  # If no 1s in the input, output is 0

        # Generate Truth Table
        st.subheader("Truth Table")
        table_data = []
        for i, bits in enumerate(product([0, 1], repeat=num_inputs)):
            row = list(bits) + [binary_output[::-1][i] if i < len(binary_output) else '0']
            table_data.append(row)

        col_names = [str(v) for v in variables] + ["Output"]
        df = pd.DataFrame(table_data, columns=col_names)
        st.dataframe(df)

        # Generate K-map
        st.subheader("Karnaugh Map")
        if num_inputs in [2, 3, 4]:  # Supports 2, 3, and 4-variable K-maps
            def generate_kmap(num_inputs, binary_output):
                if num_inputs == 2:  # 2-variable K-map (A, B)
                    kmap_size = (2, 2)
                    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
                    row_labels = ["A=0", "A=1"]
                    col_labels = ["B=0", "B=1"]
                elif num_inputs == 3:  # 3-variable K-map (A, BC)
                    kmap_size = (2, 4)
                    positions = [(0, 0), (0, 1), (0, 3), (0, 2),
                                 (1, 0), (1, 1), (1, 3), (1, 2)]
                    row_labels = ["A=0", "A=1"]
                    col_labels = ["BC=00", "BC=01", "BC=11", "BC=10"]
                elif num_inputs == 4:  # 4-variable K-map (AB, CD)
                    kmap_size = (4, 4)
                    positions = [(0, 0), (0, 1), (0, 3), (0, 2),
                                 (1, 0), (1, 1), (1, 3), (1, 2),
                                 (3, 0), (3, 1), (3, 3), (3, 2),
                                 (2, 0), (2, 1), (2, 3), (2, 2)]
                    row_labels = ["AB=00", "AB=01", "AB=11", "AB=10"]
                    col_labels = ["CD=00", "CD=01", "CD=11", "CD=10"]
                else:
                    return None

                kmap = np.zeros(kmap_size, dtype=int)
                for i, minterm in enumerate(binary_output[::-1]):
                    if minterm == '1':
                        row, col = positions[i]
                        kmap[row, col] = 1

                fig, ax = plt.subplots()
                ax.set_xticks(range(kmap.shape[1]))
                ax.set_yticks(range(kmap.shape[0]))
                ax.set_xticklabels(col_labels)
                ax.set_yticklabels(row_labels)

                # Fill grid with values
                for i in range(kmap.shape[0]):
                    for j in range(kmap.shape[1]):
                        ax.text(j, i, str(kmap[i, j]), ha="center", va="center", color="black", fontsize=14)

                ax.imshow(kmap, cmap="Blues", alpha=0.6)
                return fig

            fig = generate_kmap(num_inputs, binary_output)
            if fig:
                st.pyplot(fig)
            else:
                st.error("K-map generation for this input size is not supported.")

        # Display SOP **after** K-map
        st.subheader("Simplified Boolean Expression (SOP)")
        st.latex(f"{simplified_expr}")  # Display in LaTeX format for better readability
