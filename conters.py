# Find contours in the blue mask (hav)
contours, hierarchy = cv.findContours(maskhav, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Draw contours on a copy of the mask
maskhav_contours = np.zeros_like(maskhav)  # Create an empty mask to draw contours
cv.drawContours(maskhav_contours, contours, -1, (255, 255, 255), 2)  # White color contours
