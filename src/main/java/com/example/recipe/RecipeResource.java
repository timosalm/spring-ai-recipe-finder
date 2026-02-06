package com.example.recipe;

import io.github.bucket4j.Bucket;
import io.github.bucket4j.ConsumptionProbe;
import jakarta.servlet.http.HttpServletRequest;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/v1/recipes")
class RecipeResource {

    private static final Logger log = LoggerFactory.getLogger(RecipeResource.class);

    private final RecipeService recipeService;
    private final InputValidator inputValidator;
    private final RateLimitingConfig rateLimitingConfig;

    RecipeResource(RecipeService recipeService, InputValidator inputValidator, RateLimitingConfig rateLimitingConfig) {
        this.recipeService = recipeService;
        this.inputValidator = inputValidator;
        this.rateLimitingConfig = rateLimitingConfig;
    }

    @PostMapping("upload")
    ResponseEntity<String> addRecipeDocumentsForRag(@RequestParam("file") MultipartFile file,
                                                          @RequestParam(required = false, defaultValue = "0") int pageTopMargin,
                                                          @RequestParam(required = false, defaultValue = "0") int pageBottomMargin,
                                                          HttpServletRequest request) {
        try {
            // Rate limiting check
            String clientKey = getClientKey(request);
            Bucket bucket = rateLimitingConfig.resolveFileUploadBucket(clientKey);
            ConsumptionProbe probe = bucket.tryConsumeAndReturnRemaining(1);
            
            if (!probe.isConsumed()) {
                long waitForRefill = probe.getNanosToWaitForRefill() / 1_000_000_000;
                log.warn("Rate limit exceeded for file upload from client: {}", clientKey);
                return ResponseEntity.status(HttpStatus.TOO_MANY_REQUESTS)
                        .header("X-Rate-Limit-Retry-After-Seconds", String.valueOf(waitForRefill))
                        .body("Too many upload requests. Please try again in " + waitForRefill + " seconds.");
            }

            // Input validation
            inputValidator.validateFileUpload(file, pageTopMargin, pageBottomMargin);

            // Process the upload
            recipeService.addRecipeDocumentForRag(file.getResource(), pageTopMargin, pageBottomMargin);
            
            log.info("Recipe document uploaded successfully from client: {}", clientKey);
            return ResponseEntity.noContent().build();
        } catch (IllegalArgumentException e) {
            log.warn("Invalid file upload request: {}", e.getMessage());
            // Map validation errors to safe user-friendly messages to prevent information disclosure
            String errorMsg = e.getMessage();
            String responseMessage = "Invalid file upload request";
            
            if (errorMsg != null) {
                if (errorMsg.contains("File cannot be empty")) {
                    responseMessage = "File cannot be empty";
                } else if (errorMsg.contains("File too large")) {
                    responseMessage = "File size exceeds maximum allowed (100MB)";
                } else if (errorMsg.contains("Invalid file type") || errorMsg.contains("Only PDF files")) {
                    responseMessage = "Only PDF files are allowed";
                } else if (errorMsg.contains("Filename is required")) {
                    responseMessage = "Filename is required";
                } else if (errorMsg.contains("page") && errorMsg.contains("margin")) {
                    responseMessage = "Invalid page margin value";
                }
            }
            return ResponseEntity.badRequest().body(responseMessage);
        } catch (Exception e) {
            log.error("Error processing file upload", e);
            return ResponseEntity.internalServerError().body("Error processing file upload");
        }
    }

    /**
     * Extracts a client identifier for rate limiting (IP address).
     */
    private String getClientKey(HttpServletRequest request) {
        String xff = request.getHeader("X-Forwarded-For");
        if (xff != null && !xff.isEmpty()) {
            return xff.split(",")[0].trim();
        }
        return request.getRemoteAddr();
    }
}
