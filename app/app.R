library(shiny)
library(caret)

# Load the app-specific SVM model (10 variables, with internal scaling)
svm_model <- readRDS("svm_shiny_model.rds")

ui <- fluidPage(
  titlePanel("Breast Cancer Prediction – Tuned SVM (6 features)"),
  sidebarLayout(
    sidebarPanel(
      h4("Input Cell Characteristics"),
      
      sliderInput("radius_mean", "Radius", min = 5,  max = 30,   value = 15),
      sliderInput("texture_mean", "Texture", min = 5,  max = 45,   value = 20),
      sliderInput("area_mean", "Area", min = 140, max = 2500, value = 500),
      sliderInput("smoothness_mean", "Smoothness ", min = 0.05, max = 0.2,  value = 0.1),
      sliderInput("compactness_mean", "Compactness", min = 0.02, max = 0.3, value = 0.1),
      sliderInput("concavity_mean", "Concavity", min = 0,    max = 0.5, value = 0.1),
      
      actionButton(
        "predict_btn", "Predict Tumor Type",
        style = "color:white; background-color:#0073e6;
                 border-radius:10px; padding:10px; margin-top:20px;"
      )
    ),
    
    mainPanel(
      h3("Prediction"),
      uiOutput("prediction_ui"),
      br(),
      h5("Model: radial SVM tuned via 10-fold cross-validation on the Breast Cancer Wisconsin dataset.")
    )
  )
)

server <- function(input, output) {
  
  # Only compute prediction when button is clicked
  pred_class <- eventReactive(input$predict_btn, {
    newdata <- data.frame(
      radius_mean            = input$radius_mean,
      texture_mean           = input$texture_mean,
      area_mean              = input$area_mean,
      smoothness_mean        = input$smoothness_mean,
      compactness_mean       = input$compactness_mean,
      concavity_mean         = input$concavity_mean
    )
    
    # caret handles centering/scaling (preProcess) internally
    predict(svm_model, newdata = newdata)
  })
  
  output$prediction_ui <- renderUI({
    req(pred_class())  # wait until user has clicked
    
    if (pred_class() == "M") {
      div(
        style = "padding:20px; background:#ff4d4d; color:white; border-radius:10px;",
        h3("Prediction: MALIGNANT")
      )
    } else {
      div(
        style = "padding:20px; background:#4CAF50; color:white; border-radius:10px;",
        h3("Prediction: BENIGN")
      )
    }
  })
}

shinyApp(ui = ui, server = server)

