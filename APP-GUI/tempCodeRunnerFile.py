def display_image(self, image, canvas, title):
        """ "
        Description:
            - Plots the given (image) in the specified (canvas)
        """
        canvas.figure.clear()
        ax = canvas.figure.add_subplot(111)
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(title)
        canvas.draw()