import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Base64;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.net.URLEncoder;
import org.json.JSONObject;

public class GUI {
    private JFrame frame;
    private JLabel imageLabel;
    private JTextField questionField;
    private JButton submitButton;
    private JLabel answerLabel;
    private File selectedFile;

    public GUI() {
        frame = new JFrame("VQA Model UI");
        frame.setSize(600, 600);
        frame.setLayout(new GridBagLayout()); 
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(10, 10, 10, 10); 
        gbc.fill = GridBagConstraints.HORIZONTAL;
        
        // Image Selection
        JButton uploadButton = new JButton("Select Image");
        uploadButton.addActionListener(e -> selectImage());
        gbc.gridx = 0;
        gbc.gridy = 0;
        gbc.gridwidth = 2;
        frame.add(uploadButton, gbc);

        imageLabel = new JLabel("No Image Selected", SwingConstants.CENTER);
        imageLabel.setPreferredSize(new Dimension(400, 200));
        gbc.gridx = 0;
        gbc.gridy = 1;
        gbc.gridwidth = 2;
        frame.add(imageLabel, gbc);

        // Question Input
        JLabel questionLabel = new JLabel("Enter Question: ");
        gbc.gridx = 0;
        gbc.gridy = 2;
        gbc.gridwidth = 1;
        frame.add(questionLabel, gbc);

        questionField = new JTextField(30);
        gbc.gridx = 1;
        gbc.gridy = 2;
        frame.add(questionField, gbc);

        // Submit
        submitButton = new JButton("Ask VQA Model");
        submitButton.addActionListener(e -> sendDataToModel());
        gbc.gridx = 0;
        gbc.gridy = 3;
        gbc.gridwidth = 2;
        frame.add(submitButton, gbc);

        // Answer
        answerLabel = new JLabel("Answer: ", SwingConstants.CENTER);
        gbc.gridx = 0;
        gbc.gridy = 4;
        gbc.gridwidth = 2;
        frame.add(answerLabel, gbc);

        frame.setVisible(true);
    }

    private void selectImage() {
        JFileChooser fileChooser = new JFileChooser();
        FileNameExtensionFilter filter = new FileNameExtensionFilter("Image Files (JPG, PNG, JPEG)", "jpg", "jpeg", "png");
        fileChooser.setFileFilter(filter);

        int returnValue = fileChooser.showOpenDialog(null);
        if (returnValue == JFileChooser.APPROVE_OPTION) {
            selectedFile = fileChooser.getSelectedFile();
            
            // Update label with image preview
            try {
                BufferedImage img = ImageIO.read(selectedFile);
                ImageIcon icon = new ImageIcon(img.getScaledInstance(400, 200, Image.SCALE_SMOOTH));
                imageLabel.setIcon(icon);
            } catch (IOException e) {
                JOptionPane.showMessageDialog(frame, "Error loading image!", "Error", JOptionPane.ERROR_MESSAGE);
            }
        }
    }

    private void sendDataToModel() {
        if (selectedFile == null || questionField.getText().isEmpty()) {
            JOptionPane.showMessageDialog(frame, "Please select an image and enter a question.", "Error", JOptionPane.ERROR_MESSAGE);
            return;
        }

        String question = questionField.getText();
        try {
            byte[] imageBytes = Files.readAllBytes(Paths.get(selectedFile.getAbsolutePath()));
            String base64Image = Base64.getEncoder().encodeToString(imageBytes);
            
            String response = callVQAModel(base64Image, question);
            answerLabel.setText("Answer: " + response);
        } catch (IOException e) {
            JOptionPane.showMessageDialog(frame, "Error processing image!", "Error", JOptionPane.ERROR_MESSAGE);
        }
    }

    private String callVQAModel(String base64Image, String question) {
        try {
            HttpClient client = HttpClient.newHttpClient();
            JSONObject json = new JSONObject();
            json.put("image_base64", base64Image);
            json.put("question", question);
    
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create("http://127.0.0.1:8000/vqa/")) 
                    .header("Content-Type", "application/json") 
                    .POST(HttpRequest.BodyPublishers.ofString(json.toString(), StandardCharsets.UTF_8))
                    .build();
    
            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
    
            JSONObject jsonResponse = new JSONObject(response.body());
            return jsonResponse.getString("answer"); 
    
        } catch (Exception e) {
            e.printStackTrace();
            return "Error connecting to VQA model.";
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(GUI::new);
    }
}

