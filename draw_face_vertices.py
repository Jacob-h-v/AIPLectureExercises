import matplotlib.pyplot as plt
import numpy as np
import mplcursors
import os
# Credit: Frederik Roland
class DraggablePoints:
    def __init__(self, ax, points_data):
        self.ax = ax
        self.points = ax.scatter([], [])
        self.lines = []
        self.texts = []
        self.selected_point = None
        self.offset = (0, 0)
        self.points.set_picker(True)
        self.ax.figure.canvas.mpl_connect('pick_event', self.on_pick)
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.points_data = points_data
        self.draw()

    def draw(self):
        self.points.set_offsets(self.points_data)
        self.draw_lines()
        self.draw_point_labels()
    def draw_lines(self):
        # Clear existing lines
        for line in self.lines:
            line.remove()
        self.lines = []

        # Plot lines connecting points for each feature
        features = [
            self.points_data[:4],    # Left eyebrow
            self.points_data[4:8],  # Left eye
            self.points_data[8:12], # Right eyebrow
            self.points_data[12:16], # Right eye
            self.points_data[16:20], # Nose
            self.points_data[20:]    # Mouth
        ]
        for feature in features:
            x, y = zip(*feature)
            x = np.append(x, x[0])  # Close the loop
            y = np.append(y, y[0])  # Close the loop
            line, = self.ax.plot(x, y, color='black')
            self.lines.append(line)

    def draw_point_labels(self):
        # Clear existing points
        for text in self.texts:
            text.remove()
        self.texts = []

        for i, (x, y) in enumerate(self.points_data):
            text = self.ax.text(x, y, str(i), fontsize=12, ha='right', va='bottom', color='red')
            self.texts.append(text)
    def on_pick(self, event):
        if event.artist == self.points:
            self.selected_point = event.ind[0]
            x, y = self.points_data[self.selected_point]
            self.offset = (x - event.mouseevent.xdata, y - event.mouseevent.ydata)

    def on_motion(self, event):
        if self.selected_point is not None:
            x, y = event.xdata + self.offset[0], event.ydata + self.offset[1]
            self.points_data[self.selected_point] = (x, y)
            self.draw()
            self.ax.figure.canvas.draw()

    def on_release(self, event):
        self.selected_point = None

def save_face_to_csv(points_data, type_of_face):
    face = np.array(points_data).flatten().astype(int)
    file_exists = os.path.isfile(f'{type_of_face}.csv')
    mode = 'a' if file_exists else 'w'
    with open(f'{type_of_face}.csv', mode) as f:
        if not file_exists:
            f.write(','.join([f"P{int(i/2)}x" if i % 2 == 0 else f"P{int(i/2)}y" for i in range(len(face))]) + '\n')  # Write column headers if the file is new
        np.savetxt(f, [face], fmt='%d', delimiter=',')
    if file_exists:
        print(f"New face appended to {type_of_face}.csv")
    else:
        print(f"New face saved to {type_of_face}.csv")


def main():
    positive_face = np.array(
        [-5, 6, -3, 6, -2, 5, -3, 7, -5, 5, -3, 3, -2, 4, -3, 5, 2, 5, 3, 6, 5, 6, 3, 7, 2, 4, 3, 3, 5, 5, 3, 5, 0, 2,
         -1, -1, 0, 0, 1, -1, -3, -3, -2, -5, 2, -5, 3, -3, 1, -3, -1, -3]).reshape(-1, 2)

    fig, (ax, btn_ax1, btn_ax2) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [12, 1, 1]})
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 10)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title('Plot Face')

    ax.set_xticks(np.arange(-6, 7, 1))
    ax.set_yticks(np.arange(-6, 11, 1))

    draggable_points = DraggablePoints(ax, positive_face)

    def save_pos_face_callback(event):
        save_face_to_csv(draggable_points.points_data, 'PositiveFaces')
    def save_neg_face_callback(event):
        save_face_to_csv(draggable_points.points_data, 'NegativeFaces')
    # Add a button to save the modified face
    save_pos_button = plt.Button(btn_ax1, 'Save Positive Face')
    save_pos_button.on_clicked(save_pos_face_callback)
    save_neg_button = plt.Button(btn_ax2, 'Save Negative Face')
    save_neg_button.on_clicked(save_neg_face_callback)
    mplcursors.cursor(hover=True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
