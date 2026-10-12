// Friendly names for the types generated from the backend's OpenAPI schema
// (src/api/schema.ts -- regenerate with `npm run gen:api`, never edit by hand).
import type { components } from "./schema";

type Schemas = components["schemas"];

export type ChatEvent = Schemas["EventSchemas"]["chat"];
export type JobEvent = Schemas["EventSchemas"]["job"];

export type ChatRequest = Schemas["ChatRequest"];
export type ConversationSummary = Schemas["ConversationSummary"];
export type ConversationDetail = Schemas["ConversationDetail"];
export type Message = Schemas["Message"];
export type SourceRecord = Schemas["SourceRecord"];
export type ToolCallRecord = Schemas["ToolCallRecord"];

export type DocumentSummary = Schemas["DocumentSummary"];
export type ParseResult = Schemas["ParseResult"];
export type ParseReview = Schemas["ParseReview"];
export type TreeNode = Schemas["TreeNode"];
export type TableInfo = Schemas["TableInfo"];
export type IndexRequest = Schemas["IndexRequest"];
export type JobInfo = Schemas["JobInfo"];
export type JobDone = Schemas["JobDone"];
export type Progress = Schemas["Progress"];

export type Health = Schemas["Health"];
export type ErrorBody = Schemas["ErrorBody"];
