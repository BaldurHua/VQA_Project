import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Base64;
import java.net.URI;
import java.net.URL;
import java.net.http.*;
import java.net.http.HttpRequest.BodyPublishers;
import java.net.http.HttpResponse.BodyHandlers;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import org.json.JSONObject;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;

public class GUI {
    private JFrame frame;
    private JLabel imageLabel;
    private JTextField questionField;
    private JButton submitButton;
    private JLabel answerLabel;
    private File selectedFile;

    public GUI() {

        try {
            UIManager.setLookAndFeel("javax.swing.plaf.nimbus.NimbusLookAndFeel");
        } catch (Exception ignored) {}
        
        frame = new JFrame("Visual QA Model");
        frame.getContentPane().setBackground(new Color(245, 245, 245));
        frame.setSize(600, 500);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLocationRelativeTo(null); // Center 
        frame.setResizable(false);

        // Set Icon
        URL iconURL = getClass().getResource("/img/qa.png");
        ImageIcon icon = new ImageIcon(iconURL);
        frame.setIconImage(icon.getImage());

        JPanel panel = new JPanel();
        panel.setBackground(new Color(245, 245, 245)); 
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
        panel.setBorder(BorderFactory.createEmptyBorder(20, 40, 20, 40)); 

        // Upload Button
        JButton uploadButton = new JButton("Please Select Image");
        uploadButton.setAlignmentX(Component.CENTER_ALIGNMENT);
        uploadButton.setMaximumSize(new Dimension(200, 40));
        uploadButton.addActionListener(e -> selectImage());
        panel.add(uploadButton);
        panel.add(Box.createRigidArea(new Dimension(0, 15)));

        // Image Display 
        imageLabel = new JLabel("No image selected", SwingConstants.CENTER);
        imageLabel.setAlignmentX(Component.CENTER_ALIGNMENT);
        imageLabel.setPreferredSize(new Dimension(300, 200));
        panel.add(imageLabel);
        panel.add(Box.createRigidArea(new Dimension(0, 15)));

        // Question Field
        JLabel questionLabel = new JLabel("Enter your question:");
        questionLabel.setAlignmentX(Component.CENTER_ALIGNMENT);
        panel.add(questionLabel);
        panel.add(Box.createRigidArea(new Dimension(0, 5)));
        questionField = new JTextField();
        questionField.setMaximumSize(new Dimension(500, 30));
        panel.add(questionField);
        panel.add(Box.createRigidArea(new Dimension(0, 15)));

        // Submit Button
        submitButton = new JButton("Ask VQA Model");
        submitButton.setAlignmentX(Component.CENTER_ALIGNMENT);
        submitButton.setMaximumSize(new Dimension(160, 40));
        submitButton.addActionListener(e -> sendDataToModel());
        submitButton.setFocusPainted(false);
        panel.add(submitButton);
        panel.add(Box.createRigidArea(new Dimension(0, 20)));

        // Answer
        answerLabel = new JLabel("Answer:", SwingConstants.CENTER);
        answerLabel.setFont(new Font("Arial", Font.BOLD, 16));
        answerLabel.setAlignmentX(Component.CENTER_ALIGNMENT);
        panel.add(answerLabel);


        frame.add(panel);
        frame.setVisible(true);
    }

    private void selectImage() {
        JFileChooser fileChooser = new JFileChooser();
        JDialog chooserDialog = new JDialog(frame, "Select Image", true);
        chooserDialog.setDefaultCloseOperation(JDialog.DISPOSE_ON_CLOSE);
        URL iconURL = getClass().getResource("/img/find.png");  
        if (iconURL != null) {
            ImageIcon icon = new ImageIcon(iconURL);
            chooserDialog.setIconImage(icon.getImage());
        }
    
        chooserDialog.getContentPane().add(fileChooser);
        chooserDialog.pack();
        chooserDialog.setLocationRelativeTo(frame); 
    
        fileChooser.addActionListener(e -> {
            if (e.getActionCommand().equals(JFileChooser.APPROVE_SELECTION)) {
                selectedFile = fileChooser.getSelectedFile();
                displayImageThumbnail(selectedFile.getAbsolutePath());
            }
            chooserDialog.dispose(); 
        });
    
        chooserDialog.setVisible(true);
    }

    private void displayImageThumbnail(String path) {
        try {
            BufferedImage originalImage = ImageIO.read(new File(path));
            Image scaled = originalImage.getScaledInstance(300, 200, Image.SCALE_SMOOTH);
            ImageIcon icon = new ImageIcon(scaled);
            imageLabel.setIcon(icon);
            imageLabel.setText(""); 
        } catch (IOException e) {
            imageLabel.setText("Could not display image.");
            e.printStackTrace();
        }
    }

    private void sendDataToModel() {
        if (selectedFile == null || questionField.getText().isEmpty()) {
            JOptionPane.showMessageDialog(frame, "Please select an image and enter a question.", "Missing Input", JOptionPane.WARNING_MESSAGE);
            return;
        }

        String question = questionField.getText();
        try {
            byte[] imageBytes = Files.readAllBytes(Paths.get(selectedFile.getAbsolutePath()));
            String base64Image = Base64.getEncoder().encodeToString(imageBytes);
            String response = callVQAModel(base64Image, question);
            answerLabel.setText("Answer: " + response);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private String callVQAModel(String base64Image, String question) {
        try {
            HttpClient client = HttpClient.newHttpClient();

            String formData = "image_base64=" + URLEncoder.encode(base64Image, StandardCharsets.UTF_8) +
                              "&question=" + URLEncoder.encode(question, StandardCharsets.UTF_8);

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create("http://127.0.0.1:8000/vqa/"))
                    .header("Content-Type", "application/x-www-form-urlencoded")
                    .POST(BodyPublishers.ofString(formData))
                    .build();

            HttpResponse<String> response = client.send(request, BodyHandlers.ofString());

            if (response.statusCode() == 200) {
                JSONObject jsonResponse = new JSONObject(response.body());
                return jsonResponse.getString("answer");
            } else {
                System.out.println("Response: " + response.body());
                return "Error: Status " + response.statusCode();
            }

        } catch (Exception e) {
            e.printStackTrace();
            return "Error connecting to VQA model.";
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(GUI::new);
    }
}


