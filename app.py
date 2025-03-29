import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import symbols, SOPform
from itertools import product

st.title("Karnaugh Map Simplifier")

# Get user input for binary output
binary_output = st.text_input("Enter Binary Output (e.g., 11011000)", "")

if binary_output:
    num_inputs = (len(binary_output) - 1).bit_length()
    minterms = [i for i, bit in enumerate(binary_output[::-1]) if bit == '1']
    variables = [symbols(chr(65 + i)) for i in range(num_inputs)]
    simplified_expr = SOPform(variables, minterms)

    # Display the simplified Boolean expression
    st.write(f"**Simplified Boolean Expression:** `{simplified_expr}`")

    # Generate truth table
    st.subheader("Truth Table")
    table_data = []
    for i, bits in enumerate(product([0, 1], repeat=num_inputs)):
        row = list(bits) + [binary_output[::-1][i]]
        table_data.append(row)

    col_names = [str(v) for v in variables] + ["Output"]
    df = pd.DataFrame(table_data, columns=col_names)
    st.dataframe(df)

    # Generate K-map with labels
    if num_inputs == 3:  # Supports 3-variable K-maps (A, BC)
        st.subheader("Karnaugh Map (Labeled)")

        kmap_size = (2, 4)  # 3-variable K-map (A as row, BC as column)
        kmap = np.zeros(kmap_size, dtype=int)

        # Define Karnaugh Map positions
        positions = [(0, 0), (0, 1), (0, 3), (0, 2),
                     (1, 0), (1, 1), (1, 3), (1, 2)]

        # Map binary output to K-map
        for i, minterm in enumerate(binary_output[::-1]):
            if minterm == '1':
                row, col = positions[i]
                kmap[row, col] = 1

        # Define labels
        row_labels = ["A=0", "A=1"]
        col_labels = ["BC=00", "BC=01", "BC=11", "BC=10"]

        # Create K-map visualization
        fig, ax = plt.subplots()
        ax.set_xticks(range(kmap.shape[1]))
        ax.set_yticks(range(kmap.shape[0]))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)

        # Draw grid and labels
        for i in range(kmap.shape[0]):
            for j in range(kmap.shape[1]):
                ax.text(j, i, str(kmap[i, j]), ha="center", va="center", color="black", fontsize=14, fontweight="bold")

        ax.imshow(kmap, cmap="Blues", alpha=0.6)
        st.pyplot(fig)

    # Display SOP form
    st.subheader("Sum of Products (SOP) Form")
    st.write(f"`{simplified_expr}`")
