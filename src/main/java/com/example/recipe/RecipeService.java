package com.example.recipe;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.SafeGuardAdvisor;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.image.ImageModel;
import org.springframework.ai.image.ImagePrompt;
import org.springframework.ai.reader.pdf.PagePdfDocumentReader;
import org.springframework.ai.reader.pdf.config.PdfDocumentReaderConfig;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;
import java.util.Optional;

@Service
class RecipeService {

    private static final Logger log = LoggerFactory.getLogger(RecipeService.class);

    private final ChatClient chatClient;
    private final Optional<ImageModel> imageModel;
    private final VectorStore vectorStore;

    @Value("classpath:/prompts/recipe-for-ingredients")
    private Resource recipeForIngredientsPromptResource;

    @Value("classpath:/prompts/recipe-for-available-ingredients")
    private Resource recipeForAvailableIngredientsPromptResource;

    @Value("classpath:/prompts/prefer-own-recipe")
    private Resource preferOwnRecipePromptResource;

    @Value("classpath:/prompts/image-for-recipe")
    private Resource imageForRecipePromptResource;

    @Value("${app.available-ingredients-in-fridge}")
    private List<String> availableIngredientsInFridge;

    RecipeService(ChatClient chatClient, Optional<ImageModel> imageModel, VectorStore vectorStore) {
        this.chatClient = chatClient;
        this.imageModel = imageModel;
        this.vectorStore = vectorStore;
	}

    // ETL pipeline orchestrating the flow from raw data sources to a structured vector store
    void addRecipeDocumentForRag(Resource pdfResource, int pageTopMargin, int pageBottomMargin) {
        log.info("Add recipe document {} for rag", pdfResource.getFilename());
        var documentReaderConfig = PdfDocumentReaderConfig.builder()
                .withPageTopMargin(pageTopMargin)
                .withPageBottomMargin(pageBottomMargin)
                .build();
        // Extract: Parses PDF documents
        var documentReader = new PagePdfDocumentReader(pdfResource, documentReaderConfig);
        // Transform: Splits text into chunks based on defined token count
        var documents = new TokenTextSplitter().apply(documentReader.get());
        // Loads data into vector database
        vectorStore.accept(documents);
    }

    Recipe fetchRecipeFor(List<String> ingredients, boolean preferAvailableIngredients, boolean preferOwnRecipes) {
        Recipe recipe;
        if (!preferAvailableIngredients && !preferOwnRecipes) {
            recipe = fetchRecipeFor(ingredients);
        } else if (preferAvailableIngredients && !preferOwnRecipes) {
            recipe = fetchRecipeWithToolCallingFor(ingredients);
        } else if (!preferAvailableIngredients && preferOwnRecipes) {
            recipe = fetchRecipeWithRagFor(ingredients);
        } else {
            recipe = fetchRecipeWithRagAndToolCallingFor(ingredients);
        }

        if (imageModel.isPresent()) {
            // Only low-level API available for image models
            var imagePromptTemplate = PromptTemplate.builder()
                    .resource(imageForRecipePromptResource)
                    .variables(Map.of("recipe", recipe.name()))
                    .build();
            var imagePrompt = new ImagePrompt(imagePromptTemplate.render());
            var imageGeneration = imageModel.get().call(imagePrompt).getResult();
            return new Recipe(recipe, imageGeneration.getOutput().getUrl());
        }
        return recipe;
    }

    private Recipe fetchRecipeFor(List<String> ingredients) {
        log.info("Fetch recipe without additional information");

        return chatClient.prompt()
                .user(us -> us
                        .text(recipeForIngredientsPromptResource)
                        .param("ingredients", String.join(",", ingredients)))
                .call()
                 // Enables structured output parsing
                .entity(Recipe.class);
    }

    private Recipe fetchRecipeWithToolCallingFor(List<String> ingredients) {
        log.info("Fetch recipe with additional information from tool calling");

        return chatClient.prompt()
                .user(us -> us
                        .text(recipeForAvailableIngredientsPromptResource)
                        .param("ingredients", String.join(",", ingredients)))
                // Provides tools (methods annotated with @Tool) from this object
                .tools(this)
                .call()
                .entity(Recipe.class);
    }

    // Defines a tool
    @Tool(description = "Fetches ingredients that are available at home")
    List<String> fetchIngredientsAvailableAtHome() {
        log.info("Fetching ingredients available at home tool called by LLM");
        return availableIngredientsInFridge;
    }

    private Recipe fetchRecipeWithRagFor(List<String> ingredients) {
        log.info("Fetch recipe with additional information from vector store");
        // Configures the advisor implementing RAG
        var ragPromptTemplate = PromptTemplate.builder().resource(preferOwnRecipePromptResource).build();
        var ragSearchRequest = SearchRequest.builder().topK(2).similarityThreshold(0.7).build();
        var ragAdvisor = QuestionAnswerAdvisor.builder(vectorStore).searchRequest(ragSearchRequest).promptTemplate(ragPromptTemplate).build();

        var safeGuardAdvisor = new SafeGuardAdvisor(List.of("dump"));
         return chatClient.prompt()
                 .user(us -> us
                        .text(recipeForAvailableIngredientsPromptResource)
                        .param("ingredients", String.join(",", ingredients)))
                 // Registers the advisor implementing RAG for chat model interaction
                 .advisors(ragAdvisor)
                 .call()
                 .entity(Recipe.class);
    }

    private Recipe fetchRecipeWithRagAndToolCallingFor(List<String> ingredients) {
        log.info("Fetch recipe with additional information from vector store and tool calling");
        var ragPromptTemplate = PromptTemplate.builder().resource(preferOwnRecipePromptResource).build();
        var ragSearchRequest = SearchRequest.builder().topK(2).similarityThreshold(0.7).build();
        var ragAdvisor = QuestionAnswerAdvisor.builder(vectorStore).searchRequest(ragSearchRequest).promptTemplate(ragPromptTemplate).build();

        return chatClient.prompt()
                .user(us -> us
                        .text(recipeForAvailableIngredientsPromptResource)
                        .param("ingredients", String.join(",", ingredients)))
                .tools(this)
                .advisors(ragAdvisor)
                .call()
                .entity(Recipe.class);
    }
}